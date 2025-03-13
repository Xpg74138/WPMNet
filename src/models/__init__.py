# __init__.py
from .blocks.conv_blocks import *
from .blocks.transformer_blocks import *
from .blocks.custom_blocks import *
from .blocks.base import *
from .heads.head import *
func_blocks ={
    "Upsample":Upsample
}

# 创建一个 common 模块的字典
common_blocks = {
    "C3SPP": C3SPP,
    "SPP": SPP,
    "SPPF": SPPF,
    "Bottleneck": Bottleneck,
    "Conv": Conv,
    "CrossConv": CrossConv,
    "DWConv": DWConv,
    "DWConvTranspose2d": DWConvTranspose2d,
    "Focus": Focus,
    "ConvNextStem": ConvNextStem,
    "ConvNextBlock": ConvNextBlock,
    "Downsample": Downsample,
    "FeatureRectifyModule": FeatureRectifyModule,
    "ConvNextStem_dual": ConvNextStem_dual,
    "ConvNextBlock_dual": ConvNextBlock_dual,
    "Downsample_dual": Downsample_dual,
    "CABlock": CABlock,
    "BottleneckCSP": BottleneckCSP,
    "FeatureFusionModule": FeatureFusionModule,
    "C3": C3,
    "C3x": C3x,
    "C3Ghost": C3Ghost,
    "C3TR": C3TR,
    "PatchEmbed_dual":PatchEmbed_dual,
    "SwinTransformerBlock_dual":SwinTransformerBlock_dual,
    "dim_trans":dim_trans,
    "dim_trans_reverse":dim_trans_reverse,
    "FFl":FFl,
    "SwinTransformerBlock_dual_MY":SwinTransformerBlock_dual_MY,

}
# internal repeat
repeat_blocks={
    "BottleneckCSP": BottleneckCSP,
    "FeatureFusionModule": FeatureFusionModule,
    "C3x": C3x,
    "C3Ghost": C3Ghost,
    "C3": C3,
    "C3TR": C3TR,
}

multipleinput_blocks={
    "Concat":Concat
}

# 创建一个 head 模块的字典
heads = {
    "DecoupleHead": DecoupleHead,
    "CoupleHead": CoupleHead,
    "MY_Detect":MY_Weight_Regression_Head,
}

