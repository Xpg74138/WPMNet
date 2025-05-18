
import torch
from multitask_explainer import MultiTaskExplainer
from captum.attr import FeatureAblation, LayerConductance

# 假设你已经准备好模型和输入数据
# 请替换为你的模型加载和数据准备代码
model = ...  # 加载训练好的模型
model.eval()
device = next(model.parameters()).device

# 准备图像输入（注意 shape: (1, C, H, W)）
rgb_img = ...      # shape: (1, 3, H, W)
depth_img = ...    # shape: (1, 1, H, W)
input_tensor = torch.cat([rgb_img, depth_img], dim=1).to(device)  # shape: (1, 4, H, W)

# 初始化解释器
target_layer = 'backbone.15'
explainer = MultiTaskExplainer(model, target_layer_name=target_layer, save_dir='explain_results')

# 1️⃣ 单输入任务归因（只用 RGB / Depth 分析）
print("Step 1: 单输入任务归因")
rgb_only = torch.cat([rgb_img, torch.zeros_like(depth_img)], dim=1).to(device)
depth_only = torch.cat([torch.zeros_like(rgb_img), depth_img], dim=1).to(device)
explainer.grad_cam(rgb_only, task='classification', save_name='rgb_only_class')
explainer.grad_cam(depth_only, task='classification', save_name='depth_only_class')
explainer.grad_cam(rgb_only, task='regression', save_name='rgb_only_reg')
explainer.grad_cam(depth_only, task='regression', save_name='depth_only_reg')

# 2️⃣ 分析模态贡献（Feature Ablation）
print("Step 2: 模态贡献分析")
explainer.feature_ablation(input_tensor, task='classification', save_name='ablation_class')
explainer.feature_ablation(input_tensor, task='regression', save_name='ablation_reg')

# 可选输出通道重要性
ablator = FeatureAblation(lambda x: model(x)['classification'])
attr = ablator.attribute(input_tensor)
importance = attr.abs().mean(dim=(0, 2, 3))
print("各通道贡献值:", importance.cpu().numpy())

# 3️⃣ 多任务归因比较（Grad-CAM 对比）
print("Step 3: 多任务 Grad-CAM 对比")
explainer.compare_tasks(input_tensor, target_class=1)

# 4️⃣ 层级贡献对比（LayerConductance）
print("Step 4: 层级贡献分析")
forward_cls = lambda x: model(x)['classification']
forward_reg = lambda x: model(x)['regression']
layer = explainer.target_layer
cond_cls = LayerConductance(forward_cls, layer)
cond_reg = LayerConductance(forward_reg, layer)
score_cls = cond_cls.attribute(input_tensor, target=1).abs().mean().item()
score_reg = cond_reg.attribute(input_tensor, target=0).abs().mean().item()
print(f"该层对分类任务贡献: {score_cls:.4f}")
print(f"该层对回归任务贡献: {score_reg:.4f}")

# 5️⃣ 错误分析（预测错了怎么看错的）
print("Step 5: 错误样本分析")
with torch.no_grad():
    output = model(input_tensor)
    pred_class = output['classification'].argmax(dim=1).item()

label = 2  # 假设真实类别为 2
if pred_class != label:
    print(f"预测错误：预测为 {pred_class}，真实为 {label}")
    explainer.compare_tasks(input_tensor, target_class=pred_class)
else:
    print(f"预测正确，跳过错误分析")
