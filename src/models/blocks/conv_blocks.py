import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import autopad,drop_path
import warnings

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True,norm=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)
        self.norm = nn.BatchNorm2d(c2) if norm is True else norm if isinstance(norm, nn.Module) else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True,norm=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, p=3,g=math.gcd(c1, c2), d=d, act=act,norm=norm)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = None
        self.bias = None
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            if self.weight is None:
                self.weight = nn.Parameter(torch.ones(self.normalized_shape)).to('cuda', non_blocking=True)
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape)).to('cuda', non_blocking=True)
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            self.normalized_shape =x.shape[1:]
            if self.weight is None:
                self.weight = nn.Parameter(torch.ones(self.normalized_shape)).to('cuda', non_blocking=True)
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape)).to('cuda', non_blocking=True)
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class ConvNextBlock(nn.Module):
    def __init__(self,dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv=DWConv(c1=dim,c2=dim,k=7,act=False,norm=False)
        self.norm=LayerNorm(dim,eps=1e-6,data_format="channels_last")
        self.pwconv1=nn.Linear(dim,4*dim)
        self.act=nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,1,1)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path =DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        if self.gamma is not None:
            x = self.gamma * x
        x += input
        return x

class ConvNextStem(Conv):
    def __init__(self, c1, c2, k=4, s=4, d=1, act=False):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, p=0,g=1, d=d, act=act, norm=LayerNorm(c2,eps=1e-6))

class Downsample(nn.Module):
    def __init__(self,c1, c2, k=4, s=4, d=1, act=False,norm=False):
        super().__init__()
        self.norm=LayerNorm(c1,eps=1e-6)
        self.conv=Conv(c1,c2,k=k,s=s,p=0,act=act,norm=norm)

    def forward(self,x):
        x=self.norm(x)
        x=self.conv(x)
        return x
    
class ResNetStem(nn.Module):
    """ResNet风格的stem模块（包含初始卷积+池化）"""
    def __init__(self, c1=3, c2=64, k=7, s=2, p=None, pool=True):
        """
        Args:
            c1: 输入通道数 (默认3对应RGB图像)
            c2: 输出通道数 (默认64)
            k: 卷积核大小 (默认7)
            s: 卷积步长 (默认2)
            p: 卷积填充 (自动计算)
            pool: 是否添加最大池化层 (默认True)
        """
        super().__init__()
        # 主卷积层 (保持与ResNet原始实现一致)
        self.conv = Conv(c1, c2, k=k, s=s, p=autopad(k, p), act=True, norm=True)
        
        # 池化层 (当需要降采样时启用)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)  # 示例：224x224 -> 112x112 (当k=7,s=2,p=3)
        return self.pool(x)  # 示例：112x112 -> 56x56 (当pool=True)

class ResNetBlock(nn.Module):
    """ResNet basic block (a.k.a BasicBlock) 
    Ref: Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
    
    Args:
        c1 (int): input channels
        c2 (int): output channels
        s (int): convolution stride (for downsampling)
        expansion (int): expansion ratio (1 for BasicBlock, 4 for BottleneckBlock)
        shortcut (bool): whether to use shortcut connection
        groups (int): number of grouped convolutions
        base_width (float): width multiplier factor
    """
    expansion = 1  # BasicBlock的扩展系数
    
    def __init__(self, c1, c2, s=1, shortcut=True, groups=1, base_width=64.0):
        super().__init__()
        width = int(c2 * (base_width / 64.)) * groups
        
        # 主路径
        self.conv1 = Conv(c1, width, k=3, s=s, p=1, g=groups)
        self.conv2 = Conv(width, c2, k=3, s=1, p=1, g=groups, act=False)
        
        # 捷径路径
        self.shortcut = nn.Sequential()
        if s != 1 or c1 != c2 * self.expansion:
            if shortcut:  # 需要调整维度时使用1x1卷积
                self.shortcut = Conv(c1, c2, k=1, s=s, act=False)
            else:         # 无shortcut时保持通道一致
                self.shortcut = nn.Identity()
        
        # 最终激活函数 (与原始实现一致)
        self.act = nn.ReLU() if self.conv1.act is nn.Identity() else self.conv1.act
        
    def forward(self, x):
        # 主路径
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        
        # 残差连接 + 激活
        return self.act(x + identity)

class ResNetBottleneck(ResNetBlock):
    """Bottleneck Block for ResNet (expansion=4)"""
    expansion = 4
    
    def __init__(self, c1, c2, s=1, shortcut=True, groups=1, base_width=64.0):
        super().__init__(c1, c2, s, shortcut, groups, base_width)
        width = int(c2 * (base_width / 64.)) * groups
        
        # 重新定义主路径
        self.conv1 = Conv(c1, width, k=1, s=1)
        self.conv2 = Conv(width, width, k=3, s=s, p=1, g=groups)
        self.conv3 = Conv(width, c2 * self.expansion, k=1, act=False)
        
        # 调整shortcut路径
        if s != 1 or c1 != c2 * self.expansion:
            self.shortcut = Conv(c1, c2 * self.expansion, k=1, s=s, act=False)

class ResNetStage(nn.Module):
    """ResNet 的一个阶段（包含多个BasicBlock）"""
    def __init__(self, c1, c2, num_blocks, stride=1, groups=1, base_width=64.0):
        """
        Args:
            c1: 输入通道数
            c2: 基础通道数（实际输出通道数 = c2 * block.expansion）
            num_blocks: 当前阶段包含的块数量
            stride: 第一个块的步长（用于下采样）
        """
        super().__init__()
        # 第一个块负责下采样
        blocks = [ResNetBlock(c1, c2, s=stride, groups=groups, base_width=base_width)]
        
        # 后续块保持分辨率
        for _ in range(1, num_blocks):
            blocks.append(ResNetBlock(c2 * ResNetBlock.expansion, c2, s=1, groups=groups, base_width=base_width))
        
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)