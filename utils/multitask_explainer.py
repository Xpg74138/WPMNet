
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from datetime import datetime
from captum.attr import (
    LayerGradCam, Saliency, InputXGradient,
    FeatureAblation
)

class MultiTaskExplainer:

    # ---
    #     """初始化解释器，绑定模型、目标层，并创建保存目录和日志文件（如果提供）"""
    def __init__(self, model, target_layer_name, save_dir=None):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.target_layer_name = target_layer_name
        self.target_layer = self._find_submodule_by_name(model, target_layer_name)
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._init_log()


    # ---
    #     """通过字符串层名查找模型中的子模块，用于定位 Grad-CAM 的 hook 层"""
    def _find_submodule_by_name(self, model, layer_name):
        parts = layer_name.split('.')
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module


    # ---
    #     """构造一个前向函数，仅输出指定任务（classification 或 regression）的结果"""
    def _forward_func(self, task):
        def _func(x):
            return self.model(x)[task]
        return _func


    # ---
    #     """使用 LayerGradCam 方法对指定任务进行 Grad-CAM 归因分析，并生成可视化图像"""
    def grad_cam(self, input_tensor, task="classification", target_class=None, save_name=None):
        input_tensor = input_tensor.requires_grad_()
        gradcam = LayerGradCam(self._forward_func(task), self.target_layer)

        if task == "regression":
            target_index = 0
        else:
            with torch.no_grad():
                logits = self.model(input_tensor)['classification'][0]
                target_index = target_class if target_class is not None else logits.argmax().item()

        attr = gradcam.attribute(input_tensor, target=target_index)
        fig = self.visualize(attr[0].mean(0), input_tensor[0], title=f"{task} - class {target_index}")
        self._maybe_save(fig, "grad_cam", task, target_index, save_name)
        return fig


    # ---
    #     """使用 Saliency 方法计算输入张量的梯度归因图，并生成可视化图像"""
    def saliency(self, input_tensor, task="regression", save_name=None):
        input_tensor = input_tensor.requires_grad_()
        saliency = Saliency(self._forward_func(task))
        attr = saliency.attribute(input_tensor, target=0 if task == "regression" else None)
        fig = self.visualize(attr[0].abs().sum(0), input_tensor[0], title=f"Saliency - {task}")
        self._maybe_save(fig, "saliency", task, None, save_name)
        return fig


    # ---
    #     """使用 Input × Gradient 方法进行归因分析，反映输入对输出的乘积贡献"""
    def input_x_gradient(self, input_tensor, task="regression", save_name=None):
        input_tensor = input_tensor.requires_grad_()
        ig = InputXGradient(self._forward_func(task))
        attr = ig.attribute(input_tensor, target=0 if task == "regression" else None)
        fig = self.visualize(attr[0].abs().sum(0), input_tensor[0], title=f"InputXGradient - {task}")
        self._maybe_save(fig, "input_x_gradient", task, None, save_name)
        return fig


    # ---
    #     """使用特征遮挡（Feature Ablation）方法评估每个输入通道对模型输出的影响"""
    def feature_ablation(self, input_tensor, task="regression", save_name=None):
        ablator = FeatureAblation(self._forward_func(task))
        attr = ablator.attribute(input_tensor, target=0 if task == "regression" else None)
        fig = self.visualize(attr[0].abs().sum(0), input_tensor[0], title=f"FeatureAblation - {task}")
        self._maybe_save(fig, "feature_ablation", task, None, save_name)
        return fig


    # ---
    #     """对同一输入样本分别做分类与回归的 Grad-CAM 分析，并进行并排可视化对比"""
    def compare_tasks(self, input_tensor, target_class=None, save_name=None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        input_tensor = input_tensor.requires_grad_()

        cam_reg = LayerGradCam(self._forward_func("regression"), self.target_layer)
        attr_reg = cam_reg.attribute(input_tensor, target=0)[0].mean(0)

        cam_cls = LayerGradCam(self._forward_func("classification"), self.target_layer)
        with torch.no_grad():
            cls_logits = self.model(input_tensor)['classification'][0]
            target_index = target_class if target_class is not None else cls_logits.argmax().item()
        attr_cls = cam_cls.attribute(input_tensor, target=target_index)[0].mean(0)

        for ax, attr, title in zip(axes, [attr_reg, attr_cls], ["Regression", f"Classification ({target_index})"]):
            attr = attr.detach().cpu().numpy()
            attr = np.maximum(attr, 0)
            attr /= (attr.max() + 1e-8)
            overlay = self._overlay_mask(input_tensor[0], attr)
            ax.imshow(overlay)
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        self._maybe_save(fig, "compare_tasks", "multi", target_index, save_name)
        plt.close(fig)
        return fig


    # ---
    #     """批量处理输入数据，针对每张图像执行指定的归因方法"""
    def batch_visualize(self, input_batch, method="grad_cam", task="classification"):
        for idx, input_tensor in enumerate(input_batch):
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            self.__getattribute__(method)(input_tensor, task=task, save_name=f"{method}_{task}_sample{idx}")


    # ---
    #     """将归因结果叠加在 RGB 图像上，并绘制为可视化图像"""
    def visualize(self, heatmap_tensor, original_tensor, title=""):
        heatmap = heatmap_tensor.detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + 1e-8)

        img = original_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        img = img[..., :3]
        img -= img.min()
        img /= (img.max() + 1e-8)

        overlay = self._overlay_mask(img, heatmap)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(overlay)
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        plt.close(fig)
        return fig


    # ---
    #     """将热力图与 RGB 图像融合，生成叠加后的可视化图像（用于展示归因区域）"""
    def _overlay_mask(self, img_np, heatmap):
        cmap = plt.get_cmap('jet')(heatmap)[..., :3]
        return 0.5 * img_np + 0.5 * cmap


    # ---
    #     """将生成的图像保存到指定目录，并记录日志信息"""
    def _maybe_save(self, fig, method, task, target_class, name):
        if self.save_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name or method}_{task}_{ts}.png"
            fig_path = os.path.join(self.save_dir, filename)
            fig.savefig(fig_path)
            self._log(method, task, target_class, filename)


    # ---
    #     """初始化 CSV 日志文件，记录每次归因分析的元数据"""
    def _init_log(self):
        self.log_path = os.path.join(self.save_dir, "explain_log.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "method", "task", "target_class", "filename"])


    # ---
    #     """记录一次归因分析的信息到日志中，包括方法、任务、类别与文件名等信息"""
    def _log(self, method, task, target_class, filename):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, method, task, target_class, filename])
