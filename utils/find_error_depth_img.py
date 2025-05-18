import os
import numpy as np
import imageio

def check_zero_values(image_path, threshold=0.3):
    """
    判断图像中0值的比例是否超过阈值。

    Parameters:
    image_path (str): 图像文件的路径。
    threshold (float): 0值的比例阈值。默认为0.3，表示图像中0值的比例超过30%时输出。

    Returns:
    bool: 如果0值比例超过阈值，返回True，否则返回False。
    """
    # 读取图像
    image = imageio.imread(image_path)
    
    # 计算0值的数量和比例
    zero_count = np.sum(image == 0)
    total_pixels = image.size
    zero_ratio = zero_count / total_pixels
    
    # 如果0值比例超过阈值，返回True
    if zero_ratio > threshold:
        return True
    return False

def check_images_in_folder(folder_path, threshold=0.3):
    """
    循环读取文件夹中的图像，并打印出0值比例超过阈值的图像文件名。

    Parameters:
    folder_path (str): 存放图像的文件夹路径。
    threshold (float): 0值的比例阈值。默认为0.3，表示图像中0值的比例超过30%时输出。
    """
    # 获取文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 检查文件是否为图像文件
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 判断图像中的0值比例是否超过阈值
            if check_zero_values(file_path, threshold):
                print(f"图像 '{filename}' 中0值的比例超过了阈值。")

# 示例使用：输入文件夹路径并检查
folder_path = "/mnt/16c3c2a1-8fd2-4e6b-95ba-07a11683b744/data/700x700/val/depth"  # 修改为你的图像文件夹路径
check_images_in_folder(folder_path, threshold=0.1)  # 0.3是0值比例的阈值
