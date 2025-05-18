import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import imageio

def bicubic_interpolation(depth_image):
    """
    使用griddata替代interp2d进行插值，补全深度图像中的空洞（值为0的部分）

    Parameters:
    depth_image (numpy.ndarray): 输入的深度图像，其中的空洞部分为0值
    
    Returns:
    numpy.ndarray: 补全后的深度图像
    """
    # 获取深度图像的形状
    rows, cols = depth_image.shape

    # 创建一个掩码，表示缺失值的位置（0值）
    mask = depth_image == 0

    # 将0值替换为插值的输入点
    depth_filled = np.copy(depth_image)

    # 构建坐标网格
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # 仅选择有值的点来进行插值
    valid_points = ~mask
    valid_x = x[valid_points]
    valid_y = y[valid_points]
    valid_depth = depth_image[valid_points]

    # 使用griddata进行插值
    grid_z = scipy.interpolate.griddata(
        (valid_x, valid_y), valid_depth, (x, y), method='cubic', fill_value=0
    )

    return grid_z

depth_image = imageio.imread('/mnt/16c3c2a1-8fd2-4e6b-95ba-07a11683b744/data/700x700/val/depth/936117834228700.png')  # 读取你的深度图像文件

# 使用双三次插值补全深度图像
depth_filled = bicubic_interpolation(depth_image)

# 创建左右对比图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 显示原始带有空洞的深度图像
im0 = axes[0].imshow(depth_image, cmap='gray')
axes[0].set_title("Original Depth Image with Holes (0 values)")

# 显示补全后的深度图像
im1 = axes[1].imshow(depth_filled, cmap='gray')
axes[1].set_title("Depth Image after Bicubic Interpolation")

# 添加颜色条
fig.colorbar(im0, ax=axes[0], orientation='vertical')
fig.colorbar(im1, ax=axes[1], orientation='vertical')

# 调整布局并显示
plt.tight_layout()
plt.show()