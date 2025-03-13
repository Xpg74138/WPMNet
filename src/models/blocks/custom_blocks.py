import torch
import torch.nn as nn
from .conv_blocks import ConvNextBlock,ConvNextStem,Downsample
from .transformer_blocks import BasicLayer,PatchEmbed,Mlp,window_partition,window_reverse,PatchMerging
from timm.layers import trunc_normal_,DropPath, to_2tuple
import torch.utils.checkpoint as checkpoint

### dual_Convnext
class ConvNextStem_dual(nn.Module):
    def __init__(self, c1, c2, k=4, s=4, d=1, act=False):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__()
        self.rgb=ConvNextStem(c1-1, c2, k=k, s=s, d=d, act=act)
        self.depth=ConvNextStem(c1-3, c2, k=k, s=s, d=d, act=act)

    def forward(self,x):
        x_rgb=x[:,0:3,:,:]
        x_depth=x[:,3:,:,:]
        x_rgb=self.rgb(x_rgb)
        x_depth=self.depth(x_depth)
        return x_rgb,x_depth

class ConvNextBlock_dual(nn.Module):
    def __init__(self,dim,_,drop_path, layer_scale_init_value=1e-6):
        super().__init__()
        self.rgb = ConvNextBlock(dim, drop_path, layer_scale_init_value=layer_scale_init_value)
        self.depth = ConvNextBlock(dim, drop_path, layer_scale_init_value=layer_scale_init_value)

    def forward(self,x):
        x_rgb = x[0]
        x_depth = x[1]
        x_rgb = self.rgb(x_rgb)
        x_depth = self.depth(x_depth)
        return x_rgb, x_depth

class Downsample_dual(nn.Module):
    def __init__(self,c1, c2, k=4, s=4, d=1, act=False,norm=False):
        super().__init__()
        self.rgb = Downsample(c1, c2, k=k, s=s, d=d, act=act,norm=norm)
        self.depth = Downsample(c1, c2, k=k, s=s, d=d, act=act,norm=norm)

    def forward(self,x):
        x_rgb = x[0]
        x_depth = x[1]
        x_rgb = self.rgb(x_rgb)
        x_depth = self.depth(x_depth)
        return x_rgb, x_depth
    

class SwinTransformerBlock_dual(nn.Module):
    def __init__(self,_,embed_dim,input_resolution,depths, num_heads, drop_path,downsample,
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, fused_window_process=False,):
        super().__init__()
        self.rgb = BasicLayer(dim=embed_dim,
                                input_resolution=input_resolution,
                                depth=depths,
                                num_heads=num_heads,
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=drop_path,
                                norm_layer=norm_layer,
                                downsample=PatchMerging if downsample else None,
                                use_checkpoint=False,
                                fused_window_process=fused_window_process)
        self.depth = BasicLayer(dim=embed_dim,
                                input_resolution=input_resolution,
                                depth=depths,
                                num_heads=num_heads,
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=drop_path,
                                norm_layer=norm_layer,
                                downsample=PatchMerging if downsample else None,
                                use_checkpoint=False,
                                fused_window_process=fused_window_process)

    def forward(self,x):
        x_rgb = x[0]
        x_depth = x[1]
        x_rgb = self.rgb(x_rgb)
        x_depth = self.depth(x_depth)
        return x_rgb, x_depth

class PatchEmbed_dual(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size, img_size, norm_layer=nn.LayerNorm):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__()
        self.rgb=PatchEmbed(in_chans-3, embed_dim=embed_dim, patch_size=patch_size, img_size=img_size, norm_layer=norm_layer)
        self.depth=PatchEmbed(in_chans-3, embed_dim=embed_dim, patch_size=patch_size, img_size=img_size, norm_layer=norm_layer)

    def forward(self,x):
        x_rgb=x[:,0:3,:,:]
        x_depth=x[:,3:,:,:]
        x_rgb=self.rgb(x_rgb)
        x_depth=self.depth(x_depth)
        return x_rgb,x_depth


class dim_trans(nn.Module):
    def __init__(self,in_chans,out_chans):
        super(dim_trans, self).__init__()

    def forward(self, x):
        """
        将两个张量形状恢复。
        :param tensor1: 第一个张量，形状为 (128, 49, 768)。
        :param tensor2: 第二个张量，形状为 (128, 49, 768)。
        :return: 拼接后的张量，形状为 (128, 98, 768)。
        """
        x_rgb = x[0]
        x_depth = x[1]
        B, N, C = x_rgb.shape
        H=W=int(N**0.5)
        x_rgb = x_rgb.permute(0,2,1).reshape(B,C,H,W)
        x_depth = x_depth.permute(0, 2, 1).reshape(B, C, H, W)

        return x_rgb,x_depth

class dim_trans_reverse(nn.Module):
    def __init__(self,in_chans,out_chans):
        super(dim_trans_reverse, self).__init__()

    def forward(self, x):
        """
        将两个张量形状恢复为swin-transblock处理后的
        """
        x_rgb = x[0]
        x_depth = x[1]
        B, C, H, W = x_rgb.shape
        x_rgb = x_rgb.reshape(B,C,H*W).permute(0,2,1)
        x_depth = x_depth.reshape(B,C,H*W).permute(0,2,1)

        return x_rgb,x_depth

#米样论文中的模块

class SwinTransformerBlock_MY(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        # Independent LayerNorms for RGB and Depth streams
        self.norm1_rgb = norm_layer(dim)
        self.norm1_depth = norm_layer(dim)

        # Independent attention modules for RGB and Depth streams
        self.attn_rgb_depth = WindowAttention_CA(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # Independent MLPs for RGB and Depth streams
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.mlp_rgb = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_depth = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Drop path for stochastic depth, independent for each stream
        self.drop_path_rgb = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_depth = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Independent LayerNorms for after MLP
        self.norm2_rgb = norm_layer(dim)
        self.norm2_depth = norm_layer(dim)

        # Attention masks
        if self.shift_size > 0:
            # Create attention mask for shifted window attention
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x_rgb, x_depth):
        """
        Args:
            x_rgb: Input RGB features of shape (B, H*W, C).
            x_depth: Input Depth features of shape (B, H*W, C).
        """
        H, W = self.input_resolution
        B, L, C = x_rgb.shape
        assert L == H * W, "Input feature has wrong size"

        # === RGB Stream ===
        shortcut_rgb = x_rgb
        x_rgb = self.norm1_rgb(x_rgb)
        x_rgb = x_rgb.view(B, H, W, C)

        # === Depth Stream ===
        shortcut_depth = x_depth
        x_depth = self.norm1_depth(x_depth)
        x_depth = x_depth.view(B, H, W, C)

        # Cyclic shift for both streams if needed
        if self.shift_size > 0:
            # RGB Stream
            shifted_x_rgb = torch.roll(x_rgb, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # Depth Stream
            shifted_x_depth = torch.roll(x_depth, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x_rgb = x_rgb
            shifted_x_depth = x_depth

        # Partition windows for both streams
        x_windows_rgb = window_partition(shifted_x_rgb, self.window_size)  # nW*B, window_size, window_size, C
        x_windows_depth = window_partition(shifted_x_depth, self.window_size)  # nW*B, window_size, window_size, C

        # Flatten window partitions
        x_windows_rgb = x_windows_rgb.view(-1, self.window_size * self.window_size, C)
        x_windows_depth = x_windows_depth.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA: Attention computation for both streams
        attn_windows_rgb, attn_windows_depth = self.attn_rgb_depth(x_windows_rgb, x_windows_depth,
                                                                   mask=self.attn_mask)

        # Reshape back to window format
        attn_windows_rgb = attn_windows_rgb.view(-1, self.window_size, self.window_size, C)
        attn_windows_depth = attn_windows_depth.view(-1, self.window_size, self.window_size, C)

        # Reverse cyclic shift if needed for both streams
        if self.shift_size > 0:
            shifted_x_rgb = window_reverse(attn_windows_rgb, self.window_size, H, W)
            shifted_x_depth = window_reverse(attn_windows_depth, self.window_size, H, W)
            x_rgb = torch.roll(shifted_x_rgb, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x_depth = torch.roll(shifted_x_depth, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_rgb = window_reverse(attn_windows_rgb, self.window_size, H, W)
            x_depth = window_reverse(attn_windows_depth, self.window_size, H, W)

        # Merge back to original shape
        x_rgb = x_rgb.view(B, H * W, C)
        x_depth = x_depth.view(B, H * W, C)

        # Add residuals
        x_rgb = shortcut_rgb + self.drop_path_rgb(x_rgb)
        x_depth = shortcut_depth + self.drop_path_depth(x_depth)

        # Apply MLP for both streams
        x_rgb = x_rgb + self.drop_path_rgb(self.mlp_rgb(self.norm2_rgb(x_rgb)))
        x_depth = x_depth + self.drop_path_depth(self.mlp_depth(self.norm2_depth(x_depth)))

        return x_rgb, x_depth


class FFl(nn.Module):
    #Features_Fusion_layer
    def __init__(self,in_chans,out_chans):
        super(FFl, self).__init__()

    def forward(self, x):
        x_rgb=x[0]
        x_depth=x[1]
        x=x_rgb+x_depth
        return x

class WindowAttention_CA(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted windows.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w],indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # Separate qkv and proj for both streams
        self.qkv_rgb = nn.Linear(dim, dim * 3, bias=qkv_bias)  # For RGB stream
        self.qkv_depth = nn.Linear(dim, dim * 3, bias=qkv_bias)  # For Depth stream

        self.proj_rgb = nn.Linear(dim, dim)  # For RGB stream
        self.proj_depth = nn.Linear(dim, dim)  # For Depth stream

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop_rgb = nn.Dropout(proj_drop)
        self.proj_drop_depth = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_rgb, x_depth, mask=None):
        """
        Args:
            x_rgb: input features of the RGB stream, shape (num_windows*B, N, C)
            x_depth: input features of the Depth stream, shape (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x_rgb.shape

        # Calculate qkv for the RGB stream
        qkv_rgb = self.qkv_rgb(x_rgb).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_rgb, k_rgb, v_rgb = qkv_rgb[0], qkv_rgb[1], qkv_rgb[2]

        # Calculate qkv for the Depth stream
        qkv_depth = self.qkv_depth(x_depth).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_depth, k_depth, v_depth = qkv_depth[0], qkv_depth[1], qkv_depth[2]

        # Swap the Q values between the two streams
        q_rgb, q_depth = q_depth.clone(), q_rgb.clone()

        # Apply scaling factor to q1 and q2
        q_rgb = q_rgb * self.scale
        q_depth = q_depth * self.scale

        # Compute attention for both streams
        attn_rgb = (q_rgb @ k_rgb.transpose(-2, -1))
        attn_depth = (q_depth @ k_depth.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_rgb = attn_rgb + relative_position_bias.unsqueeze(0)
        attn_depth = attn_depth + relative_position_bias.unsqueeze(0)

        # Apply softmax and dropout
        if mask is not None:
            nW = mask.shape[0]
            attn_rgb = attn_rgb.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn_depth = attn_depth.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn_rgb = attn_rgb.view(-1, self.num_heads, N, N)
            attn_depth = attn_depth.view(-1, self.num_heads, N, N)

        attn_rgb = self.softmax(attn_rgb)
        attn_depth = self.softmax(attn_depth)

        attn_rgb = self.attn_drop(attn_rgb)
        attn_depth = self.attn_drop(attn_depth)

        # Multiply attention weights with values (v_rgb, v_depth)
        x_rgb = (attn_rgb @ v_rgb).transpose(1, 2).reshape(B_, N, C)
        x_depth = (attn_depth @ v_depth).transpose(1, 2).reshape(B_, N, C)

        # Final linear projection and dropout for each stream
        x_rgb = self.proj_rgb(x_rgb)
        x_depth = self.proj_depth(x_depth)

        x_rgb = self.proj_drop_rgb(x_rgb)
        x_depth = self.proj_drop_depth(x_depth)

        return x_rgb, x_depth

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class BasicLayer_MY(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self,dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_MY(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x_rgb,x_depth=x[0],x[1]
                x_rgb,x_depth = blk(x_rgb,x_depth)
        if self.downsample is not None:
            x_rgb = self.downsample(x_rgb)
            x_depth = self.downsample(x_depth)
        return x_rgb,x_depth

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class SwinTransformerBlock_dual_MY(nn.Module):
    def __init__(self,_,embed_dim,input_resolution,depths, num_heads, drop_path,downsample,
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, fused_window_process=False,):
        super().__init__()
        self.m = BasicLayer_MY(dim=embed_dim,
                                input_resolution=input_resolution,
                                depth=depths,
                                num_heads=num_heads,
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=drop_path,
                                norm_layer=norm_layer,
                                downsample=PatchMerging if downsample else None,
                                use_checkpoint=False,
                                fused_window_process=fused_window_process)


    def forward(self,x):
        x_rgb,x_depth = self.m(x)
        return x_rgb,x_depth