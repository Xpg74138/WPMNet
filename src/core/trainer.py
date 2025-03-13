from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import mlflow
import albumentations as A
from albumentations.pytorch import ToTensorV2
import hydra
from hydra.utils import instantiate
import os
import time
import inspect
# 自定义模块导入
from .builder import XModel  
from .datasets import CustomDataset  # 数据集类
from .loss import PWLoss, PWAPLoss # 损失函数
from .metrics import  MetricsCalculator  # 评估指标
from ..utils.visualization import ResultsVisualizer  # 导入可视化模块
from ..utils.labelnorm import DataNormalizer

class GenericTrainer:
    def __init__(self, cfg: DictConfig):
        """
        初始化训练器
        Args:
            cfg: 通过Hydra加载的配置
        """
        # 设置设备和随机种子
        device_name = cfg.training.parameter.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_name)
        torch.manual_seed(cfg.training.parameter.seed)

        # 保存完整配置
        self.cfg = cfg
        
        # 初始化组件
        self._build_components()
        
        # 初始化MLflow 
        self._setup_mlflow()
        
        # 初始化可视化工具
        self.visualizer = ResultsVisualizer(self.cfg.data.class_names)
        
    def _build_components(self):
        """构建训练所需的所有组件"""
        # 1. 构建模型 - 使用现有方法，因为模型可能需要特殊处理
        self.model = self._build_model(self.cfg.model)
        
        # 2. 构建数据加载器
        self.train_loader, self.val_loader = self._build_dataloaders(self.cfg)
        
        # 3. 构建优化器 - 使用Hydra实例化
        self.optimizer = instantiate(
            self.cfg.optimizer, 
            params=self.model.parameters()
        )
        
        # 4. 构建学习率调度器
        self.scheduler = self._build_scheduler(self.cfg.scheduler)
        
        # 5. 构建损失函数 - 使用配置或返回默认值
        self.criterion = self._build_criterion(self.cfg.training.task)
        
        # 6. 构建指标计算器
        self.metrics_calculator = self._build_metrics(self.cfg.training.metrics)
        
        # 7. 创建混合精度训练的GradScaler（如果启用）
        self.use_amp = self.cfg.training.parameter.get('use_amp', False)
        self.scaler = torch.amp.GradScaler('cuda',enabled=self.use_amp)



    def _setup_mlflow(self):
        """设置MLflow实验跟踪"""


        # 使用上下文管理器启动运行
        if mlflow.active_run():
            self.mlflow_run = mlflow.active_run()
        else:
            # 否则创建新的运行
            self.mlflow_run = mlflow.start_run(
                run_name=self.cfg.experiment.get('name', f"run_{time.strftime('%Y%m%d_%H%M%S')}"),
                nested=self.cfg.mlflow.get('nested', False)
            )
        
        # 记录关键参数
        mlflow.log_params({
            "model": str(self.cfg.model._target_),
            "optimizer": str(self.cfg.optimizer._target_),
            "batch_size": self.cfg.training.parameter.batch_size,
            "epochs": self.cfg.training.parameter.epochs,
            "learning_rate": self.cfg.optimizer.lr,
            "seed": self.cfg.training.parameter.seed,
        })
        
        # 记录所有标签
        if 'tags' in self.cfg.experiment:
            for tag in self.cfg.experiment.tags:
                mlflow.set_tag(tag, True)
                
        # 记录配置文件
        mlflow.log_dict(OmegaConf.to_container(self.cfg, resolve=True), "config.yaml")

    def _build_model(self, model_cfg: DictConfig) -> nn.Module:
        """模型构建"""
        if model_cfg.task == "weight":
            model = XModel(model_cfg.Model, task=model_cfg.task)
        elif model_cfg.task == "weight_posture":
            model = XModel(model_cfg.Model, task=model_cfg.task)
        else:
            raise ValueError(f"不支持的模型类型: {model_cfg.task}")
        
        model = model.to(self.device)
        return model

    def _build_dataloaders(self, cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
        """构建数据加载器"""
        # 构建数据增强转换
        transform = self._build_transforms(cfg.augmentation)
        
        # 构建数据集
        
        train_set = CustomDataset(
            cfg.data.train.path,
            cfg.data.class_names,
            cfg.model.type,
            img_size=cfg.data.img_size,
            transform=transform,
            cache_file=cfg.data.train.cache
        )
        val_set = CustomDataset(
            cfg.data.val.path,
            cfg.data.class_names,
            cfg.model.type,
            img_size=cfg.data.img_size,
            transform=transform,
            cache_file=cfg.data.val.cache
        )

        #使用采样器创建数据加载器
        train_loader = instantiate(
            cfg.data.train_loader,
            dataset=train_set,
            batch_size=cfg.data.train_loader.batch_size,
            shuffle=True,
            num_workers=cfg.data.train_loader.num_workers,
            pin_memory=True,
            prefetch_factor=3,  # 每个 worker 预加载的批次数
            persistent_workers=True,  # 保持 workers 进程运行
        )
        
        val_loader = instantiate(
            cfg.data.val_loader,
            dataset=val_set,
            batch_size=cfg.data.val_loader.batch_size,
            shuffle=False,
            num_workers=cfg.data.val_loader.num_workers,
            pin_memory=True,
            prefetch_factor=3,  # 每个 worker 预加载的批次数
            persistent_workers=True,  # 保持 workers 进程运行
        )
        
        return train_loader, val_loader

    def _build_transforms(self, aug_cfg: DictConfig) -> A.Compose:
        """Albumentations数据增强管道（支持多模态）"""
        transforms = []
        
        # 随机裁剪配置
        if hasattr(aug_cfg, 'random_crop'):
            transforms.append(
                A.RandomResizedCrop(
                    size=[aug_cfg.random_crop.crop_size,aug_cfg.random_crop.crop_size],
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.33),
                    p=aug_cfg.random_crop.prob
                )
            )
        
        # 颜色抖动配置
        if hasattr(aug_cfg, 'color_jitter'):
            transforms.append(
                A.ColorJitter(
                    brightness=aug_cfg.color_jitter.get('jitter_brightness', 0.0),
                    contrast=aug_cfg.color_jitter.get('jitter_contrast', 0.0),
                    saturation=aug_cfg.color_jitter.get('jitter_saturation', 0.0),
                    hue=aug_cfg.color_jitter.get('jitter_hue', 0.0),
                    always_apply=False,
                    p=aug_cfg.color_jitter.prob
                )
            )
        
        # 水平翻转
        if hasattr(aug_cfg, 'random_horizontal_flip'):
            transforms.append(
                A.HorizontalFlip(p=aug_cfg.random_horizontal_flip.prob)
            )
        if hasattr(aug_cfg, 'random_vertical_flip'):
            transforms.append(
                A.VerticalFlip(p=aug_cfg.random_vertical_flip.prob)
            )

        # 随机旋转
        if hasattr(aug_cfg, 'random_rotation'):
            transforms.append(
                A.Rotate(limit=aug_cfg.random_rotation.degrees, p=aug_cfg.random_rotation.prob)
            )
        
        # 基础变换（同步应用）
        base_transforms = [
            # RGB标准化
            A.Normalize(
                mean=[0.5215,0.5253,0.0218],
                std=[0.2128,0.2199,0.0358],
                max_pixel_value=255.0,
                p=1.0
            ),
            # Depth标准化（如果需要）
            A.Normalize(
                mean=[0.0211], 
                std=[0.0357],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(transpose_mask=True)
        ]
        
        # 组合所有变换
        # 检查配置中是否有多模态设置
        modality = self.cfg.data.get('modality', ['rgb'])
        additional_targets = {}
        
        if 'depth' in modality:
            additional_targets['depth'] = 'image'
            
        transform_pipeline = A.Compose(
            transforms + base_transforms,
            additional_targets=additional_targets
        )
        
        return transform_pipeline

    def _build_optimizer(self, opt_cfg: DictConfig) -> optim.Optimizer:
        """优化器构建"""
        optimizer_class = getattr(optim, opt_cfg._target_.split('.')[-1])
        
        # 移除_target_键
        opt_params = {k: v for k, v in opt_cfg.items() if k != '_target_'}
        return optimizer_class(self.model.parameters(), **opt_params)

    def _build_scheduler(self, sched_cfg: DictConfig):
        """
        构建学习率调度器
        
        Args:
            sched_cfg: 调度器配置，如果为 None 或 type 为 'none' 则不使用调度器
        
        Returns:
            学习率调度器或 None
        """
        # 检查是否使用调度器
        if sched_cfg is None or not hasattr(sched_cfg, '_target_') or sched_cfg._target_.lower() == 'none':
            print("No learning rate scheduler will be used.")
            return None
        
        # 获取调度器类名
        scheduler_class_name = sched_cfg._target_.split('.')[-1]
        scheduler_class = getattr(optim.lr_scheduler, scheduler_class_name)
        
        # 获取调度器的构造函数参数
        sig = inspect.signature(scheduler_class)
        params = sig.parameters.keys()
        valid_params = [p for p in params if p not in ['self', 'optimizer']]
        
        # 从配置中提取有效参数
        kwargs = {}
        for param in valid_params:
            if hasattr(sched_cfg, param):
                kwargs[param] = getattr(sched_cfg, param)
        
        return scheduler_class(self.optimizer, **kwargs)

    def _build_criterion(self, task_cfg: DictConfig) -> nn.Module:
        """损失函数构建"""
        # 如果提供了显式损失函数配置
        if hasattr(task_cfg, 'loss') and hasattr(task_cfg.loss, '_target_'):
            return instantiate(task_cfg.loss)
        
        # 否则使用默认损失函数
        if task_cfg.type == "weight":
            return PWLoss()
        elif task_cfg.type == "weight_posture":
            return PWAPLoss()
        else:
            raise ValueError(f"不支持的任务类型: {task_cfg.type}")

    def _build_metrics(self, metrics_cfg: DictConfig) -> MetricsCalculator:
        """构建指标计算器"""
        metrics_config = {}
        
        # 从配置中读取各任务类型的指标
        if hasattr(metrics_cfg, "regression"):
            metrics_config["regression"] = metrics_cfg.regression
        if hasattr(metrics_cfg, "classification"):
            metrics_config["classification"] = metrics_cfg.classification
        
        return MetricsCalculator(metrics_config)

    def train_epoch(self, epoch: int):
        """训练一个周期"""
        # 创建数据标准化器
        self.data_normalizer = DataNormalizer(self.train_loader.dataset.labels_weight)
        self.model.train()
        total_loss = 0.0
        epoch_start_time = time.time()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = {k: (self.data_normalizer.normalize(v).to(self.device) if k == 'regression' else v.to(self.device)) for k, v in targets.items()}

            # 使用自动混合精度
            with torch.amp.autocast('cuda',enabled=self.use_amp):
                # 前向传播
                outputs = self.model(images)
                loss, loss_items = self.criterion(outputs, targets)
            
            # 反向传播（使用scaler进行缩放）
            self.optimizer.zero_grad()
    

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # 记录指标
            total_loss += loss.item()
            
            # 记录到MLflow（仅在指定间隔和主进程）
            if batch_idx % self.cfg.visualization.get('log_interval', 10) == 0:
                mlflow.log_metric("training/batch_loss", loss.item(),
                                 step=epoch*len(self.train_loader)+batch_idx)
                
                # 记录学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                mlflow.log_metric("training/learning_rate", current_lr,
                                 step=epoch*len(self.train_loader)+batch_idx)
        
        # 计算周期平均损失和时间
        epoch_loss = total_loss / len(self.train_loader)


        epoch_time = time.time() - epoch_start_time
        
        
        mlflow.log_metric("training/epoch_loss", epoch_loss, step=epoch)
        mlflow.log_metric("training/epoch_time_seconds", epoch_time, step=epoch)
        
        return epoch_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证模型性能"""
        self.model.eval()
        val_loss = 0.0
        all_preds = {}
        all_targets = {}

        # 保存一批样本用于特征可视化
        sample_images = None
        
        for batch_idx, (images, targets) in enumerate(self.val_loader):
            # 移动数据到设备
            images = images.to(self.device)
            targets = {k: (self.data_normalizer.normalize(v).to(self.device) if k == 'regression' else v.to(self.device)) for k, v in targets.items()}
            
            # 保存第一批图像用于特征可视化
            if batch_idx == 0:
                sample_images = images.clone()

            # 使用自动混合精度
            with torch.amp.autocast('cuda',enabled=self.use_amp):
                # 模型输出多个任务的预测结果
                outputs = self.model(images)
                loss, loss_items = self.criterion(outputs, targets)
                val_loss += loss.item()

            outputs = {k: (self.data_normalizer.denormalize(v) if k == 'regression' else v) for k, v in outputs.items()}
            targets = {k: (self.data_normalizer.denormalize(v) if k == 'regression' else v) for k, v in targets.items()}
            # 收集每个任务的预测和目标
            for task_type in outputs.keys():
                if task_type not in all_preds:
                    all_preds[task_type] = []
                    all_targets[task_type] = []
                
                all_preds[task_type].append(outputs[task_type].cpu())
                all_targets[task_type].append(targets[task_type].cpu())   
        
        # 合并预测结果
        predictions = {
            task_type: torch.cat(preds) 
            for task_type, preds in all_preds.items()
        }
        ground_truth = {
            task_type: torch.cat(targets)
            for task_type, targets in all_targets.items()
        }
        
        # 计算所有指标
        metrics = self.metrics_calculator.compute_all(predictions, ground_truth)
        epoch_loss = val_loss / len(self.val_loader)
        mlflow.log_metric("val/loss", epoch_loss, step=epoch)
        # 记录到MLflow（更清晰的层次结构）
        for task_type, task_metrics in metrics.items():
            for metric_name, value in task_metrics.items():
                mlflow.log_metric(f"val/{task_type}/{metric_name}", value, step=epoch)
        
        # 每隔N个周期生成可视化结果
        vis_interval = self.cfg.visualization.get('visualization_interval', 5)
        if epoch % vis_interval == 0 or epoch == self.cfg.training.parameter.epochs - 1:
            # 寻找可视化的目标层
            target_layer = self.cfg.visualization.get('target_layer', None)
            
            try:
                # 生成并记录可视化结果
                self.visualizer.log_to_mlflow(
                    predictions=predictions, 
                    targets=ground_truth, 
                    epoch=epoch,
                    model=self.model,
                    sample_images=sample_images,
                    target_layer=target_layer
                )
            except Exception as e:
                print(f"生成可视化结果失败: {str(e)}")
                # 记录错误但继续训练
                mlflow.log_text(str(e), f"visualization/error_epoch_{epoch}.txt")

        return metrics

    def run(self):
        """运行完整的训练流程"""
        try:
            best_metric = float('inf')  # 假设是最小化指标，如果是最大化，使用float('-inf')
            tracking_metric = self.cfg.get('tracking', {}).get('metric', 'val/regression/mae')
            
            # 训练循环
            for epoch in range(self.cfg.training.parameter.epochs):
                # 记录周期开始（仅在主进程）

                mlflow.log_metric("epoch", epoch, step=epoch)
                
                # 训练和验证
                train_loss = self.train_epoch(epoch)
                val_metrics = self.validate(epoch)
                
                # 解析嵌套指标字典获取跟踪指标值
                current_metric = None
                metric_parts = tracking_metric.split('/')
                if len(metric_parts) == 3:  # 例如 val/regression/mae
                    part1, part2, part3 = metric_parts
                    if part2 in val_metrics and part3 in val_metrics[part2]:
                        current_metric = val_metrics[part2][part3]
                
                # 如果跟踪指标有效，检查是否保存最佳模型（仅在主进程）
                if current_metric is not None :
                    is_best = current_metric < best_metric  # 对于最小化指标
                    
                    if is_best:
                        best_metric = current_metric
                        self._save_checkpoint(epoch, "best_model.pth")
                        mlflow.log_metric("best_epoch", epoch, step=epoch)
                        mlflow.log_metric("best_metric", best_metric, step=epoch)
                    
                # 保存最新模型（仅在主进程）
                self._save_checkpoint(epoch, "latest_model.pth")
                
                # 更新学习率
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_metric)
                    else:
                        self.scheduler.step()
        finally:
            if mlflow.active_run():
                mlflow.end_run()

    def _save_checkpoint(self, epoch: int, filename: str):
        """保存检查点"""
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        output_dir = hydra_cfg['runtime']['output_dir']
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": OmegaConf.to_container(self.cfg),
        }
        
        # 只有在使用调度器时才保存调度器状态
        if self.scheduler is not None:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        # 记录到MLflow
        if self.cfg.mlflow.get('log_artifacts', True):
            mlflow.log_artifact(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型和优化器状态
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        # 如果有调度器且检查点中包含调度器状态，则加载调度器状态
        if self.scheduler is not None and "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        # 返回上次训练的轮次
        return checkpoint["epoch"]
