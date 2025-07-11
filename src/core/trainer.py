import os
import time
import inspect
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import mlflow
import albumentations as A
from albumentations.pytorch import ToTensorV2
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LinearLR, SequentialLR

# 自定义模块导入
from .builder import XModel
from .datasets import CustomDataset  # 数据集类
from .loss import PWLoss, PWAPLoss  # 损失函数
from .metrics import MetricsCalculator  # 评估指标
from ..utils.visualization import ResultsVisualizer
from ..utils.labelnorm import DataNormalizer
from.augumentation import RandomBackgroundReplacement,DualNormalize

class GenericTrainer:
    def __init__(self,
                 cfg: DictConfig,
                 local_rank: int = 0,
                 is_main_process: bool = True,
                 ddp_enabled: bool = False):
        
        self.cfg = cfg
        self.local_rank = local_rank
        self.is_main_process = is_main_process
        self.ddp_enabled = ddp_enabled
        
        # 设置设备
        if ddp_enabled:
            # DDP：每个进程都只使用 local_rank 对应的 GPU
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            # 单机单卡(或CPU)
            device_name = cfg.training.parameter.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            self.device = torch.device(device_name)

        # 设置随机种子
        torch.manual_seed(cfg.training.parameter.seed)

        # 构建组件
        self._build_components()
        
        # 只有主进程才进行mlflow配置
        if self.is_main_process:
            self._setup_mlflow()

        # 可视化工具（主进程才需要可视化和log）
        if self.is_main_process:
            self.visualizer = ResultsVisualizer(self.cfg.data.class_names)
        else:
            self.visualizer = None
        
    def _build_components(self):
        # 1. 构建模型
        self.model = self._build_model(self.cfg.model)

        # 2. 构建数据加载器（含分布式Sampler）
        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = self._build_dataloaders(self.cfg)
        
        # 3. 构建优化器
        self.optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())
        
        # 4. 构建调度器
        self.scheduler = self._build_scheduler(self.cfg.scheduler)
        
        # 5. 构建损失函数
        self.criterion = self._build_criterion(self.cfg.training.task)
        
        # 6. 构建指标计算器
        self.metrics_calculator = self._build_metrics(self.cfg.training.metrics)
        
        # 7. 混合精度
        self.use_amp = self.cfg.training.parameter.get('use_amp', True)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # 8. 如果是分布式，就封装DDP
        if self.ddp_enabled:
            # 注意，需要先 to(self.device)，再封装为 DDP
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )

    def _setup_mlflow(self):
        # 如果已经在外部 start_run() 了，这里可以不再start，而是直接记录
        if mlflow.active_run():
            self.mlflow_run = mlflow.active_run()
        else:
            self.mlflow_run = mlflow.start_run(
                run_name=self.cfg.experiment.get('name', f"run_{time.strftime('%Y%m%d_%H%M%S')}"),
                nested=self.cfg.mlflow.get('nested', False)
            )
        
        mlflow.log_params({
            "model": str(self.cfg.model._target_),
            "optimizer": str(self.cfg.optimizer._target_),
            "batch_size": self.cfg.training.parameter.batch_size,
            "epochs": self.cfg.training.parameter.epochs,
            "learning_rate": self.cfg.optimizer.lr,
            "seed": self.cfg.training.parameter.seed,
        })
        
        if 'tags' in self.cfg.experiment:
            for tag in self.cfg.experiment.tags:
                mlflow.set_tag(tag, True)

        mlflow.log_dict(OmegaConf.to_container(self.cfg, resolve=True), "config.yaml")

    def _build_model(self, model_cfg: DictConfig) -> nn.Module:
        if model_cfg.task == "weight":
            model = XModel(model_cfg.Model, task=model_cfg.task)
        elif model_cfg.task == "weight_posture":
            model = XModel(model_cfg.Model, task=model_cfg.task)
        else:
            raise ValueError(f"不支持的模型类型: {model_cfg.task}")
        
        model = model.to(self.device)
        model.log_info()
        return model

    def _build_dataloaders(self, cfg: DictConfig):
        transform ,val_transform= self._build_transforms(cfg.augmentation)
        
        
        train_set = CustomDataset(
            cfg.data.train.path,
            cfg.data.class_names,
            cfg.model.type,
            img_size=cfg.data.img_size,
            transform=transform,
            cache_file=cfg.data.train.cache,
            input_channel=cfg.model.Model.input_channel
        )
        
        val_set = CustomDataset(
            cfg.data.val.path,
            cfg.data.class_names,
            cfg.model.type,
            img_size=cfg.data.img_size,
            transform=val_transform,
            cache_file=cfg.data.val.cache,
            input_channel=cfg.model.Model.input_channel
        )

        # 分布式采样器
        if self.ddp_enabled:
            train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=False)
            val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=False)
            shuffle_flag = False  # sampler 自己控制 shuffle，不要再由 DataLoader 控制
        else:
            train_sampler = None
            val_sampler = None
            shuffle_flag = True  # 只有单GPU时，DataLoader内部shuffle

        train_loader = instantiate(
            cfg.data.train_loader,
            dataset=train_set,
            batch_size=cfg.data.train_loader.batch_size,
            sampler=train_sampler,
            shuffle=shuffle_flag,
            num_workers=cfg.data.train_loader.num_workers,
            pin_memory=True,
            prefetch_factor=3,
            persistent_workers=True,
        )
        
        val_loader = instantiate(
            cfg.data.val_loader,
            dataset=val_set,
            batch_size=cfg.data.val_loader.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=cfg.data.val_loader.num_workers,
            pin_memory=True,
            prefetch_factor=3,
            persistent_workers=True,
        )
        
        return train_loader, val_loader, train_sampler, val_sampler

    def _build_transforms(self, aug_cfg: DictConfig) -> A.Compose:
        transforms = []
        additional_targets = {}  # 提前初始化

        # 1. 先注册所有additional targets
        modality = self.cfg.data.get('modality', ['rgb'])
        if 'depth' in modality:
            additional_targets['depth'] = 'image'
        if hasattr(aug_cfg, 'random_background_replacement'):
            additional_targets['mask'] = 'mask'  # 关键修正

        if hasattr(aug_cfg, 'random_background_replacement'):
            transforms.append(
                RandomBackgroundReplacement(
                    p=aug_cfg.random_background_replacement.prob
                    )
            )
        if hasattr(aug_cfg, 'random_crop'):
            transforms.append(
                A.RandomResizedCrop(
                    height=aug_cfg.random_crop.crop_size,
                    width=aug_cfg.random_crop.crop_size,
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.33),
                    p=aug_cfg.random_crop.prob
                )
            )
        if hasattr(aug_cfg, 'color_jitter'):
            transforms.append(
                A.ColorJitter(
                    brightness=aug_cfg.color_jitter.get('jitter_brightness', 0.0),
                    contrast=aug_cfg.color_jitter.get('jitter_contrast', 0.0),
                    saturation=aug_cfg.color_jitter.get('jitter_saturation', 0.0),
                    hue=aug_cfg.color_jitter.get('jitter_hue', 0.0),
                    p=aug_cfg.color_jitter.prob
                )
            )
        if hasattr(aug_cfg, 'random_horizontal_flip'):
            transforms.append(
                A.HorizontalFlip(p=aug_cfg.random_horizontal_flip.prob)
            )
        if hasattr(aug_cfg, 'random_vertical_flip'):
            transforms.append(
                A.VerticalFlip(p=aug_cfg.random_vertical_flip.prob)
            )
        if hasattr(aug_cfg, 'random_rotation'):
            transforms.append(
                A.Rotate(limit=aug_cfg.random_rotation.degrees, p=aug_cfg.random_rotation.prob)
            )

            
        # 基础变换
        base_transforms = [
            # DualNormalize(
            #     rgb_mean=(0.52898648,0.5133086, 0.52153534),
            #     rgb_std=(0.25797648,0.25973946,0.26773506),
            #     depth_mean=0.0208,
            #     depth_std=0.0150552,
            #     p=1.0
            # ),
            DualNormalize(
                rgb_mean=(0.52898648,0.5133086, 0.52153534),
                rgb_std=(0.25797648,0.25973946,0.26773506),
                depth_min=0,
                depth_max=255,
                p=1.0
            ),
            ToTensorV2(transpose_mask=True)
        ]
        
            
        transform_pipeline = A.Compose(
            transforms + base_transforms,
            additional_targets=additional_targets
        )
        
        transform_pipeline_val = A.Compose(
            base_transforms,
            additional_targets=additional_targets
        )
        return transform_pipeline,transform_pipeline_val

    from torch.optim.lr_scheduler import LinearLR, SequentialLR

    def _build_scheduler(self, sched_cfg: DictConfig):
        # 不用 scheduler 的情况保持不变
        if sched_cfg is None or not hasattr(sched_cfg, '_target_') or sched_cfg._target_.lower() == 'none':
            print("No learning rate scheduler will be used.")
            return None

        # 1) 创建主调度器
        scheduler_class_name = sched_cfg._target_.split('.')[-1]
        scheduler_class = getattr(optim.lr_scheduler, scheduler_class_name)
        sig = inspect.signature(scheduler_class)
        main_kwargs = {
            k: getattr(sched_cfg, k)
            for k in sig.parameters.keys()
            if k not in ('self', 'optimizer') and hasattr(sched_cfg, k)
        }
        main_scheduler = scheduler_class(self.optimizer, **main_kwargs)

        # 2) 如果没有 warmup，直接返回主调度器
        if not hasattr(sched_cfg, 'warmup_iters'):
            return main_scheduler

        # 3) 创建 warmup 调度器（线性从 warmup_start_factor 到 1.0）
        warmup_iters = int(sched_cfg.warmup_iters)
        start_factor = float(getattr(sched_cfg, 'warmup_start_factor', 0.01))
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_iters
        )

        # 4) 串联成一个 SequentialLR，milestones 在 warmup_iters
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_iters]
        )
        return scheduler


    def _build_criterion(self, task_cfg: DictConfig) -> nn.Module:
        if hasattr(task_cfg, 'loss') and hasattr(task_cfg.loss, '_target_'):
            return instantiate(task_cfg.loss)
        
        if task_cfg.type == "weight":
            return PWLoss()
        elif task_cfg.type == "weight_posture":
            return PWAPLoss()
        else:
            raise ValueError(f"不支持的任务类型: {task_cfg.type}")

    def _build_metrics(self, metrics_cfg: DictConfig) -> MetricsCalculator:
        metrics_config = {}
        if hasattr(metrics_cfg, "regression"):
            metrics_config["regression"] = metrics_cfg.regression
        if hasattr(metrics_cfg, "classification"):
            metrics_config["classification"] = metrics_cfg.classification
        
        return MetricsCalculator(metrics_config)

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        # 如果使用分布式sampler，需要设置一下 epoch，保证shuffle的seed不同
        if self.ddp_enabled and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        self.data_normalizer = DataNormalizer(self.train_loader.dataset.labels_weight)
        self.model.train()
        total_loss = 0.0
        total_cls_loss=0.0
        total_reg_loss=0.0
        epoch_start_time = time.time()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = {
                k: (
                    self.data_normalizer.normalize(v).to(self.device)
                    if k == 'regression' else v.to(self.device)
                )
                for k, v in targets.items()
            }
            #with torch.autograd.set_detect_anomaly(True):
            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                outputs = self.model(images)
                loss, loss_items = self.criterion(outputs, targets)
            
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                #gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            if len(loss_items)>2:
                total_cls_loss += loss_items[0].item()
                total_reg_loss += loss_items[1].item()
            total_loss += loss.item()

            # 只有在主进程才记录日志
            if self.is_main_process and (batch_idx % self.cfg.visualization.get('log_interval', 10) == 0):
                step = epoch * len(self.train_loader) + batch_idx
                mlflow.log_metric("training/batch_loss", loss.item(), step=step)
                current_lr = self.optimizer.param_groups[0]['lr']
                mlflow.log_metric("training/learning_rate", current_lr, step=step)
            del outputs, loss, loss_items
        epoch_loss = total_loss / len(self.train_loader)
        epoch_cls_loss= total_cls_loss / len(self.train_loader)
        epoch_reg_loss= total_reg_loss / len(self.train_loader)
        epoch_time = time.time() - epoch_start_time

        if self.is_main_process:
            mlflow.log_metric("training/epoch_loss", epoch_loss, step=epoch)
            mlflow.log_metric("training/epoch_cls_loss", epoch_cls_loss, step=epoch)
            mlflow.log_metric("training/epoch_reg_loss", epoch_reg_loss, step=epoch)
            mlflow.log_metric("training/epoch_time_seconds", epoch_time, step=epoch)
        
        return epoch_loss


    def validate(self, epoch: int):
        self.model.eval()
        val_loss = 0.0
        val_cls_loss=0.0
        val_reg_loss=0.0
        all_preds = {}
        all_targets = {}

        # 如果使用分布式sampler，需要保证相同
        if self.ddp_enabled and self.val_sampler is not None:
            self.val_sampler.set_epoch(epoch)

        sample_images = None
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                images = images.to(self.device)
                targets = {
                    k: (
                        self.data_normalizer.normalize(v).to(self.device)
                        if k == 'regression' else v.to(self.device)
                    )
                    for k, v in targets.items()
                }
                if batch_idx == 0:
                    sample_images = images.clone()

                with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                    outputs = self.model(images)
                    loss, loss_items = self.criterion(outputs, targets)
                    val_loss += loss.item()
                    if len(loss_items)>2:
                        val_cls_loss += loss_items[0].item()
                        val_reg_loss += loss_items[1].item()

                # 把分布后的输出先反归一化
                outputs = {
                    k: (
                        self.data_normalizer.denormalize(v) if k == 'regression' else v
                    ) for k,v in outputs.items()
                }
                targets = {
                    k: (
                        self.data_normalizer.denormalize(v) if k == 'regression' else v
                    ) for k,v in targets.items()
                }

                for task_type in outputs.keys():
                    if task_type not in all_preds:
                        all_preds[task_type] = []
                        all_targets[task_type] = []
                    all_preds[task_type].append(outputs[task_type].cpu())
                    all_targets[task_type].append(targets[task_type].cpu())

            epoch_loss = val_loss / len(self.val_loader)
            epoch_cls_loss= val_cls_loss / len(self.val_loader)
            epoch_reg_loss= val_reg_loss / len(self.val_loader)

        # -----------------------------
        # (可选) 下面是简单做法：仅在 rank=0 上记录日志和可视化
        #       如果需要严格的汇总(例如拼接预测向量)，则需要对 all_preds/all_targets 做分布式 gather
        # -----------------------------
        if self.ddp_enabled:
            # 对 val_loss 先做一个 all_reduce
            total_val_loss = torch.tensor(epoch_loss, device=self.device)
            total_val_cls_loss = torch.tensor(epoch_cls_loss, device=self.device)
            total_val_reg_loss = torch.tensor(epoch_reg_loss, device=self.device)
            dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_val_cls_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_val_reg_loss, op=dist.ReduceOp.SUM)
            # 计算平均
            world_size = dist.get_world_size()
            epoch_loss = (total_val_loss / world_size).item()
            epoch_cls_loss = (total_val_cls_loss / world_size).item()
            epoch_reg_loss = (total_val_reg_loss / world_size).item()

        # 只有主进程做度量计算并log
        if self.is_main_process:
            # 这里暂且只用 rank=0 的 all_preds 做演示
            # 如果想要全部进程的预测，就要做 gather
            predictions = {
                t: torch.cat(p) for t,p in all_preds.items()
            }
            ground_truth = {
                t: torch.cat(g) for t,g in all_targets.items()
            }
            metrics = self.metrics_calculator.compute_all(predictions, ground_truth)

            mlflow.log_metric("val/loss", epoch_loss, step=epoch)
            mlflow.log_metric("val/cls_loss", epoch_cls_loss, step=epoch)
            mlflow.log_metric("val/reg_loss", epoch_reg_loss, step=epoch)
            for task_type, task_metrics in metrics.items():
                for metric_name, value in task_metrics.items():
                    mlflow.log_metric(f"val/{task_type}/{metric_name}", value, step=epoch)

            # 可视化
            vis_interval = self.cfg.visualization.get('visualization_interval', 5)
            if epoch % vis_interval == 0 or epoch == self.cfg.training.parameter.epochs - 1:
                target_layer = self.cfg.visualization.get('target_layer', None)
                if self.visualizer is not None:
                    try:
                        self.visualizer.log_to_mlflow(
                            predictions=predictions, 
                            targets=ground_truth, 
                            epoch=epoch,
                            model=self.model.module if self.ddp_enabled else self.model,
                            sample_images=sample_images,
                            target_layer=target_layer
                        )
                        del sample_images
                    except Exception as e:
                        print(f"生成可视化结果失败: {str(e)}")
                        mlflow.log_text(str(e), f"visualization/error_epoch_{epoch}.txt")

            return metrics
        else:
            # 非主进程就简单返回
            return {}

    def run(self):
        try:
            best_metric = float('inf')
            tracking_metric = self.cfg.get('tracking', {}).get('metric', 'val/regression/mae')
            
            for epoch in range(self.cfg.training.parameter.epochs):
                if self.is_main_process:
                    mlflow.log_metric("epoch", epoch, step=epoch)
                
                train_loss = self.train_epoch(epoch)
                val_metrics = self.validate(epoch)
                
                # 在 rank=0 上根据 val_metrics 检查是否最佳
                if self.is_main_process:
                    current_metric = None
                    metric_parts = tracking_metric.split('/')
                    if len(metric_parts) == 3:
                        part1, part2, part3 = metric_parts
                        if part2 in val_metrics and part3 in val_metrics[part2]:
                            current_metric = val_metrics[part2][part3]
                    
                    current_f1=val_metrics["classification"]["f1"]

                    if current_metric is not None:
                        is_best = current_metric < best_metric
                        if is_best:
                            best_metric = current_metric
                            self._save_checkpoint(epoch, "best_model.pth")
                            mlflow.log_metric("best_epoch", epoch, step=epoch)
                            mlflow.log_metric("best_metric", best_metric, step=epoch)
                    try:
                        if self.cfg.snapshot:
                            if  self.cfg.snapshot==4 and current_f1>=0.8 and current_f1<0.85:
                                self._save_checkpoint(epoch, f"model_{0.8}.pth")
                                self.cfg.snapshot=self.cfg.snapshot-1
                            elif  self.cfg.snapshot==3 and current_f1>=0.85 and current_f1<0.90:
                                self._save_checkpoint(epoch, f"model_{0.85}.pth")
                                self.cfg.snapshot=self.cfg.snapshot-1
                            elif  self.cfg.snapshot==2 and current_f1>=0.90 and current_f1<0.95:
                                self._save_checkpoint(epoch, f"model_{0.90}.pth")
                                self.cfg.snapshot=self.cfg.snapshot-1
                            elif  self.cfg.snapshot==1 and current_f1>=0.95 and current_f1<1.0:
                                self._save_checkpoint(epoch, f"model_{0.95}.pth")
                                self.cfg.snapshot=self.cfg.snapshot-1
                    except:
                        pass
                    
                    self._save_checkpoint(epoch, "latest_model.pth")

                # 分布式时，各进程都要 step scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) and self.is_main_process:
                        # 只在主进程调用 step
                        self.scheduler.step(current_metric)
                    elif not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()

        finally:
            if self.is_main_process and mlflow.active_run():
                mlflow.end_run()

    def _save_checkpoint(self, epoch: int, filename: str):
        # 只在主进程保存模型
        if not self.is_main_process:
            return

        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        output_dir = hydra_cfg['runtime']['output_dir']
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # 如果是DDP，需要保存model.module的state_dict
        model_to_save = self.model.module if self.ddp_enabled else self.model

        checkpoint = {
            "epoch": epoch,
            "model_state": model_to_save.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": OmegaConf.to_container(self.cfg),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        
        if self.cfg.mlflow.get('log_artifacts', True):
            mlflow.log_artifact(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.ddp_enabled:
            self.model.module.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint["model_state"])
        
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        if self.scheduler is not None and "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        return checkpoint["epoch"]