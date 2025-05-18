import cv2
import numpy as np
from scipy.ndimage import binary_dilation

def fill_depth_holes(depth_img, max_iter=20):
    """基于邻域传播的空洞填充算法"""
    # 创建有效像素掩膜（0表示空洞）
    valid_mask = (depth_img > 0).astype(np.uint8)
    
    # 初始化填充图像
    filled = depth_img.copy()
    
    # 结构元素用于控制膨胀方向
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    # 迭代填充
    for _ in range(max_iter):
        # 找到空洞边缘（无效像素周围的轮廓）
        edges = cv2.morphologyEx(valid_mask, cv2.MORPH_GRADIENT, kernel)
        
        # 没有更多空洞需要填充时退出
        if np.sum(edges) == 0:
            break
        
        # 计算邻域中位数（保留边缘特征）
        median_neighbors = cv2.medianBlur(filled, 3)
        
        # 仅更新空洞边缘像素
        filled[(edges > 0) & (valid_mask == 0)] = median_neighbors[(edges > 0) & (valid_mask == 0)]
        
        # 更新有效像素掩膜
        valid_mask = (filled > 0).astype(np.uint8)
    
    return filled

def process_depth_image(png_path):
    # 读取原始16位深度图
    raw_depth = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)
    
    if raw_depth is None:
        print("Error: Failed to read depth image")
        return

    # 执行空洞填充（关键步骤）
    filled_depth = fill_depth_holes(raw_depth)
    
    # 保存补全后的16位深度图（保留原始数值范围）
    cv2.imwrite("filled_depth_16bit.png", filled_depth.astype(np.uint16))
    
    # 可视化部分（可选）
    valid_mask = filled_depth > 0
    valid_pixels = filled_depth[valid_mask]
    
    # 动态归一化显示
    min_depth = np.percentile(valid_pixels, 2)
    max_depth = np.percentile(valid_pixels, 98)
    
    normalized = np.clip((filled_depth.astype(np.float32) - min_depth) / (max_depth - min_depth), 0, 1)
    vis_8bit = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(vis_8bit, cv2.COLORMAP_JET)
    #colored[~valid_mask] = [0, 0, 255]  # 红色标记残留空洞
    
    cv2.imwrite("vis_uint8.png", colored.astype(np.uint8))
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_depth_image("/media/xpg/Part1/pen23/2023-07-22/Depth/512406082346400.png")