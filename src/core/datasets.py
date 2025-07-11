from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import cv2
from PIL import ImageFile
from skimage.transform import resize as rs
import skimage.io as io
from src.core.augumentation import RandomBackgroundReplacement
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomDataset(Dataset):
    def __init__(self, list_path, class_names, model_type,input_channel,img_size=416, transform=None, cache_file=None,seg_aug=True):
        self.img_size = img_size
        self.transform = transform
        self.class_names = class_names
        self.cache_file = cache_file
        self.model_type=model_type
        self.cache_data=[]
        self.cache_mask=[]
        self.seg_files=None
        self.input_channel = input_channel
        # 读取文件路径
        with open(list_path, "r") as file:
            self.files = file.readlines()
        if model_type=="weight_posture":
            rgb_img_files, depth_img_files, labels_weight, labels_posture = zip(
                *(obj.split(',') for obj in tqdm(self.files))
            )
        else:
            rgb_img_files, depth_img_files, labels_weight,_ = zip(
                *(obj.split(',') for obj in tqdm(self.files))
            )
        if any(isinstance(t, RandomBackgroundReplacement) for t in self.transform.transforms):
            seg_file=list_path.replace(".txt","_seg.txt")
            with open(seg_file, "r") as file:
                seg_files = file.readlines()
                self.seg_files = [line.strip() for line in seg_files]
            
        # 将 zip 对象转换为列表
        self.rgb_img_files = list(rgb_img_files)
        self.depth_img_files = list(depth_img_files)
        self.labels_weight = list(map(float, labels_weight))
        if model_type=="weight_posture":
            self.labels_posture = [lp.strip() for lp in labels_posture]

        # 如果选择预加载，则将所有图像加载到内存中
        if cache_file is None:
            self.rgb_images = [self.load_image(f, mode='RGB') for f in tqdm(self.rgb_img_files, desc="Loading RGB images")]
            self.depth_images = [self.load_image(f, mode='I', dtype='uint16') for f in tqdm(self.depth_img_files, desc="Loading Depth images")]
        elif cache_file:
            if os.path.exists(cache_file):
                self.cache_data= torch.load(cache_file,weights_only=False)
            else:
                self.preprocess_and_cache()

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

    def preprocess_and_cache(self):
        """将图像预处理并缓存到指定目录"""
        for idx in tqdm(range(len(self.rgb_img_files)), desc="Caching images"):
            rgb_img = self.load_image(self.rgb_img_files[idx], mode='RGB')
            depth_img = self.load_image(self.depth_img_files[idx], mode='I', dtype='uint16')
            # 将深度图像扩展为3通道
            # depth_img = np.repeat(np.expand_dims(depth_img, axis=-1), 3, axis=2)

            # 将图像数据添加到缓存列表
            self.cache_data.append((rgb_img, depth_img))

        # 保存所有图像数据到一个大文件
        torch.save(self.cache_data, self.cache_file)

    def __getitem__(self, index):
        # 如果预加载，则从内存中获取图像
        if self.cache_file is None:
            rgb_img = self.rgb_images[index]
            depth_img = self.depth_images[index]
        elif self.cache_file:
            # 从缓存中加载
            rgb_img, depth_img = self.cache_data[index]
        else:
            # 否则从磁盘中加载
            rgb_img = self.load_image(self.rgb_img_files[index], mode='RGB', dtype='uint8')
            depth_img = self.load_image(self.depth_img_files[index], mode='I', dtype='uint8')
            # depth_img = np.repeat(np.expand_dims(depth_img, axis=-1), 3, axis=2)

        # 如果有 transform，则对图像进行处理
        depth_img = np.expand_dims(depth_img, axis=-1)
        if self.transform:
            if self.seg_files != None:
                augmented = self.transform(image=rgb_img, depth=depth_img,mask_path=self.seg_files[index])
            else:
                augmented = self.transform(image=rgb_img, depth=depth_img)
            rgb_img = augmented['image']  # 形状变为 (C,H,W)
            depth_img = augmented['depth']  # 形状变为 (C,H,W)
        
        if self.input_channel == 1:
            # 如果输入通道为1，输入为depth图像
            image_input = depth_img
        elif self.input_channel == 3:   
            # 如果输入通道为3，则确保 RGB 图像保持三通道
            image_input = rgb_img
        else:
            # 合并 RGB 和深度图像，维度：[4, img_size, img_size]
            image_input = torch.cat([rgb_img, depth_img[0:1,:,:]], dim=0)

        # 获取标签
        # 修改返回格式
        if self.model_type == "weight_posture":
            class_label = self.class_names.index(self.labels_posture[index])
            # 返回字典格式的标签
            targets = {
                'classification': torch.tensor([class_label], dtype=torch.float32),
                'regression': torch.tensor([self.labels_weight[index]], dtype=torch.float32)
            }
        else:
            # 仅体重预测任务
            targets = {
                'regression': torch.tensor([self.labels_weight[index]], dtype=torch.float32)
            }

        return image_input, targets

    def __len__(self):
        return len(self.files)

