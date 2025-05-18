import cv2
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform,DualTransform
from typing import Any, Optional
import random
from PIL import Image
import io

class RandomBackgroundReplacement(ImageOnlyTransform):
    def __init__(self, p: float = 0.5,img_size:int =224,blur_kernel_size: int = 27, alpha: float = 0.7):
        """
        随机背景替换增强
        
        Args:
            always_apply (bool): 是否总是应用该变换
            p (float): 应用概率
        """
        super().__init__(p=p)
        self.blur_kernel = blur_kernel_size
        self.alpha = alpha
        self._logged = False
        self.img_size=img_size

    def __eq__(self,other):
        # 比较全部属性
        return self.__dict__ == other.__dict__

    def apply(self, img: np.ndarray, mask_path: Optional[np.ndarray] = None, **params: Any) -> np.ndarray:
        """应用背景替换到图像"""
        if mask_path is None:
            print("⚠️ No mask provided")
            return img
        if img.shape[-1] != 3:
            return img
        
        mask = self.load_seg_mask(mask_path)

        # 校验并预处理mask
        if len(mask.shape) != 2:
            raise ValueError("Mask must be a 2D array")
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

         # 1. 处理mask
        mask = self._process_mask(mask)
        
        # 2. 生成半透明背景层 (随机颜色/纹理)
        overlay = self._generate_overlay(img.shape)
        
        # 3. 边缘羽化
        blend_mask = self._feather_mask(mask)
        
        # 4. 创建背景区域的布尔掩码 (扩展为3通道)
        background_area = (blend_mask < 255)  # 形状 (H, W)
        background_area = np.repeat(background_area[..., np.newaxis], 3, axis=2)  # 扩展为 (H, W, 3)
        
        # 5. 混合图像 (仅修改背景区域)
        result = img.copy()
        result[background_area] = (
            img[background_area] * (1 - self.alpha) + 
            overlay[background_area] * self.alpha
        ).astype(np.uint8)

        # 记录增强样本
        # if not self._logged:
        #     self._log_augmentation(img, result, mask)
            
        return result
    
    def load_seg_mask(self,mask_path):
        """安全加载分割掩膜
        Args:
            seg_path: 掩膜文件路径 (支持png/jpg等格式)
        Returns:
            np.ndarray: 单通道uint8数组 (h, w)
        """
        try:
            # 灰度模式读取（确保单通道）
            mask = self.load_image(mask_path, mode='I', dtype='uint8')
            if mask is None:
                raise ValueError(f"无法读取文件: {mask_path}")
            
            # 格式校验
            assert len(mask.shape) == 2, "掩膜应为二维数组"
            assert mask.dtype == np.uint8, "数据类型应为uint8"
            
            return mask
        except Exception as e:
            print(f"加载掩膜失败: {mask_path} | 错误: {str(e)}")
            raise  # 根据需求选择抛出异常或返回空数组

    def load_image(self, path, mode, dtype=None):
        """安全加载图像（自动关闭文件句柄）"""
        try:
            with Image.open(path) as img:  # 使用上下文管理器
                img = img.convert(mode).resize((self.img_size, self.img_size))
                return np.array(img, dtype=dtype)
        except Exception as e:  # 明确捕获异常类型
            try:
                img = io.imread(path)
                return img.astype(dtype) if dtype else img
            except Exception as e:
                print(f"加载图像失败: {path} | 错误: {str(e)}")
                raise  # 或返回 None/默认值

    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        """确保mask为二值uint8格式"""
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return np.where(mask > 127, 255, 0).astype(np.uint8)

    def _feather_mask(self, mask: np.ndarray) -> np.ndarray:
        """羽化边缘（仅羽化背景过渡区域）"""
        blurred = cv2.GaussianBlur(mask, (self.blur_kernel, self.blur_kernel), 0)
        return np.where(mask == 255, 255, blurred)  # 保持前景不羽化

    def _generate_overlay(self, shape: tuple) -> np.ndarray:
        """生成随机半透明层"""
        # 随机选择生成模式
        mode = random.choice(["color", "gradient", "noise"])
        
        if mode == "color":
            # 随机纯色层
            overlay = np.array([[random.randint(0, 255) for _ in range(3)]], dtype=np.uint8)
            overlay = np.tile(overlay, (shape[0], shape[1], 1))
            
        elif mode == "gradient":
            # 双色渐变
            color1 = np.array([random.randint(0, 255) for _ in range(3)])
            color2 = np.array([random.randint(0, 255) for _ in range(3)])
            h, w = shape[:2]
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(3):
                overlay[..., i] = np.linspace(color1[i], color2[i], w).astype(np.uint8)
                
        else:  # noise
            # 噪点纹理
            overlay = np.random.randint(0, 255, shape, dtype=np.uint8)
            overlay = cv2.blur(overlay, (31, 31))
            
        return overlay

    def _log_augmentation(self, orig_img: np.ndarray, aug_img: np.ndarray, mask: np.ndarray):
        """可视化增强效果并记录到MLflow"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(orig_img)
        axes[0].set_title("Original", fontsize=10)
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Mask", fontsize=10)
        axes[1].axis('off')
        
        axes[2].imshow(aug_img)
        axes[2].set_title("Augmented", fontsize=10)
        axes[2].axis('off')

        plt.tight_layout()
        mlflow.log_figure(fig, "augmentation_demo.png")
        plt.close(fig)
        self._logged = True

    @property
    def targets_as_params(self):
        return ["mask_path"]

    def get_params_dependent_on_data(self, params: dict,data:dict) -> dict:
        """从数据中获取mask参数"""
        return {"mask_path": data["mask_path"]}

    def get_transform_init_args_names(self) -> tuple:
        return ("always_apply", "p")
    
class DualNormalize(DualTransform):
    """
    同时处理RGB和Depth图像的归一化变换
    RGB使用3通道均值/标准差，Depth使用单通道均值/标准差
    """
    def __init__(self, 
                 rgb_mean=(0.5215, 0.5253, 0.0218),
                 rgb_std=(0.2128, 0.2199, 0.0358),
                 depth_min=500,
                 depth_max=8000,
                 max_pixel_value=255.0,
                 always_apply=True,
                 p=1.0):
        super().__init__(always_apply, p)
        self.rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(rgb_std, dtype=np.float32)
        self.depth_min = np.array(depth_min, dtype=np.float32)
        self.depth_max = np.array(depth_max, dtype=np.float32)
        self.max_pixel_value = max_pixel_value

    def apply(self, img, **params):
        """
        img: 输入的图像数据 (H,W,C)
        params: 包含target_name信息，用于判断当前处理的是rgb还是depth
        """
        
        if img.shape[-1] == 3:  # RGB图像
            mean = self.rgb_mean
            std = self.rgb_std
        else:  # Depth图像
            img = self.normalize_depth(img, self.depth_min, self.depth_max)
            return img
            # mean = self.depth_mean
            # std = self.depth_std
        
        # 确保均值和std的维度与输入匹配
        if len(img.shape) == 2:  # 单通道 (H,W)
            img = np.expand_dims(img, -1)
        
        # 归一化计算
        normalized = (img.astype(np.float32) / self.max_pixel_value - mean) / std
        
        # 保持原始通道数
        if img.shape[-1] == 1:
            return normalized[:, :, 0] if len(img.shape) == 3 else normalized
        return normalized
    
    def normalize_depth(self,depth_raw, depth_min=500, depth_max=8000):
        # depth_raw: np.ndarray, dtype=np.uint16, shape=(H, W)
        depth = depth_raw.astype(np.float32)  # 转换为 float32 类型以便处理
        depth = np.clip(depth, depth_min, depth_max)  # 将深度值限制在 [depth_min, depth_max] 范围内
        depth = (depth - depth_min) / (depth_max - depth_min)  # 归一化到 [0, 1]
        return depth

    def get_transform_init_args_names(self):
        return ("rgb_mean", "rgb_std", "depth_mean", "depth_std", "max_pixel_value")