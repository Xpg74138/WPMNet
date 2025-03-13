import torch
import torch.nn as nn
from blocks.conv_blocks import Conv

class DecoupledHead(nn.Module):
    def __init__(self, ch=256, nc=80, anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors)   # number of anchors
        self.merge = Conv(ch, 256, 1, 1)
        self.cls_convs1 = Conv(256, 256, 3, 1, 1)
        self.cls_convs2 = Conv(256, 256, 3, 1, 1)
        self.reg_convs1 = Conv(256, 256, 3, 1, 1)
        self.reg_convs2 = Conv(256, 256, 3, 1, 1)
        self.weight_convs1=Conv(512, 256, 3, 1, 1)
        self.weight_convs2 = Conv(256, 256, 3, 1, 1)
        self.cls_preds = nn.Conv2d(256, self.nc * self.na, 1)  # 一个1x1的卷积，把通道数变成类别数，比如coco 80类（主要对目标框的类别，预测分数）
        self.reg_preds = nn.Conv2d(256, 4 * self.na, 1)  # 一个1x1的卷积，把通道数变成4通道，因为位置是xywh
        self.obj_preds = nn.Conv2d(256, 1 * self.na, 1)  # 一个1x1的卷积，把通道数变成1通道，通过一个值即可判断有无目标（置信度）
        self.weight_preds = nn.Conv2d(256, 1 * self.na, 1)

    def forward(self, x):
        x = self.merge(x)
        x1 = self.cls_convs1(x)
        x1 = self.cls_convs2(x1)
        x11 = self.cls_preds(x1)
        x2 = self.reg_convs1(x)
        x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        x22 = self.obj_preds(x2)
        x3 = torch.cat((x, x1), dim=1)
        x3 = self.weight_convs1(x3)
        x3 = self.weight_convs2(x3)
        x3 = self.weight_preds(x3)
        out = torch.cat([x21, x22, x3,x11], 1)  # 把分类和回归结果按channel维度，即dim=1拼接
        return out


class Decoupled_Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=0, inplace=True):  # detection layer
        super().__init__()

        self.nc = nc  # number of classes
        self.no = nc + 6  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors)# number of anchors
        self.grid = torch.zeros(1)  # init grid
        anchors = torch.tensor(anchors).float().view(-1, 2)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))  #这个形状有点像输出  # init anchor grid
        self.register_buffer('anchors', anchors)  # shape(nl,na,2)
        self.m = DecoupledHead(ch, nc, anchors)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.stride=None

    def forward(self, x,img_size):
        stride = img_size // x.size(2)
        self.stride = stride
        z = []  # inference output
        x = self.m(x)  # conv
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:  # inference
            if self.onnx_dynamic or self.grid.shape[2:4] != x.shape[2:4]:
                self.grid= self._make_grid(nx, ny)

            # y = x.sigmoid()
            # if self.inplace:
            #     y[..., 0:2] = (y[..., 0:2] * 2 + self.grid) * self.stride  # xy
            #     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid  # wh
            # else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
            #     xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
            #     xy = (xy * 2 + self.grid) * self.stride  # xy
            #     wh = (wh * 2) ** 2 * self.anchor_grid  # wh
            #     y = torch.cat((xy, wh, conf), 4)
            # # z.append(y.view(bs, -1, self.no))
            # y=y.view(bs, -1, self.no)
            #
            if self.inplace:
                x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
                x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
                #x[..., 4:] = x[..., 4:].sigmoid() # conf, cls
                x[..., 4] = x[..., 4].sigmoid()  # conf, cls#注意这里改为了【x,y,w,h,conf,weight,classes】
                #x[..., 5] = x[..., 5].sigmoid()*200 #尝试将值体重预测值限制在0-200
                x[..., 6:] = x[..., 6:].sigmoid()
            else:
                x[..., 0:2] = (x[..., 0:2] + self.grid) * stride  # xy
                x[..., 2:4] = x[..., 2:4] ** 2 * (4 * self.anchor_grid)  # wh
            x = x.view(bs, -1, self.no)
        # return x if self.training else (torch.cat(z, 1),) if self.export else y
        return x

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20) -> torch.Tensor:
        """
        Create a grid of (x, y) coordinates

        :param nx: Number of x coordinates
        :param ny: Number of y coordinates
        """
        d = 'cuda'
        yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    # def _make_grid(self, nx=20, ny=20):
    #     d = self.anchors.device
    #     t = self.anchors.dtype
    #     shape = 1, self.na, ny, nx, 2  # grid shape
    #     y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
    #     if torch.__version__>='1.10.0':  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
    #         yv, xv = torch.meshgrid(y, x, indexing='ij')
    #     else:
    #         yv, xv = torch.meshgrid(y, x)
    #     grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    #     anchor_grid = (self.anchors * self.stride).view((1, self.na, 1, 1, 2)).expand(shape)
    #     return grid, anchor_grid