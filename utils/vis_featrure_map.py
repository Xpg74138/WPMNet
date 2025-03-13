from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

model = resnet18().eval()
# 获取输入图像
img = read_image("img.png")
# 预处理以适应所选模型
input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

with SmoothGradCAMpp(model, 'layer4') as cam_extractor:
    # 预处理数据并将其馈送给模型
    out = model(input_tensor.unsqueeze(0))
    # 通过指定类索引和模型输出来获取 CAM
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)


# 可视化原始 CAM
plt.imshow(activation_map[0].squeeze(0).numpy())
plt.axis('off')
plt.tight_layout()
plt.show()

# 缩放 CAM 并叠加它
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# 显示结果
plt.imshow(result)
plt.axis('off')
plt.tight_layout()
plt.show()