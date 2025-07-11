import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import pandas as pd
import numpy as np

# 导入你已有的模块
from src.core.builder import XModel
from src.core.datasets import CustomDataset
from src.utils.labelnorm import DataNormalizer
from src.core.metrics import MetricsCalculator
from src.core.augumentation import DualNormalize
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

@hydra.main(config_path="configs", config_name="test_config.yaml")
def test(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 构建模型
    model = XModel(cfg.model.Model, task=cfg.model.task)
    model = model.to(device)
    model.eval()

    # 2. 加载模型参数
    checkpoint_path = cfg.testing.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    # 3. 构建测试数据集和 DataLoader
    transform = A.Compose([
        DualNormalize(
            rgb_mean=(0.52898648, 0.5133086, 0.52153534),
            rgb_std=(0.25797648, 0.25973946, 0.26773506),
            depth_min=0,
            depth_max=255,
            p=1.0
        ),
        ToTensorV2(transpose_mask=True)
    ], additional_targets={'depth': 'image'} if 'depth' in cfg.data.modality else {})

    test_set = CustomDataset(
        cfg.data.test.path,
        cfg.data.class_names,
        cfg.model.type,
        img_size=cfg.data.img_size,
        transform=transform,
        cache_file=cfg.data.test.cache,
        input_channel=cfg.model.Model.input_channel
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.data.test_loader.batch_size,
        shuffle=False,
        num_workers=cfg.data.test_loader.num_workers,
        pin_memory=True
    )

    # 4. 初始化归一化器和评估指标
    data_normalizer = DataNormalizer(test_set.labels_weight)
    metrics_calculator = MetricsCalculator(cfg.testing.metrics)

    # 5. 模型推理与评估
    all_preds, all_targets = {}, {}
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = {
                k: (
                    data_normalizer.normalize(v).to(device)
                    if k == 'regression' else v.to(device)
                ) for k, v in targets.items()
            }

            outputs = model(images)
            outputs = {
                k: (
                    data_normalizer.denormalize(v) if k == 'regression' else v
                ) for k, v in outputs.items()
            }
            targets = {
                k: (
                    data_normalizer.denormalize(v) if k == 'regression' else v
                ) for k, v in targets.items()
            }

            for task_type in outputs.keys():
                all_preds.setdefault(task_type, []).append(outputs[task_type].cpu())
                all_targets.setdefault(task_type, []).append(targets[task_type].cpu())

    # 6. 计算并打印评估指标
    predictions = {t: torch.cat(p) for t, p in all_preds.items()}
    ground_truth = {t: torch.cat(g) for t, g in all_targets.items()}
    trueL=np.array(ground_truth["regression"].squeeze())
    preL=np.array(predictions["regression"].squeeze())
    df = pd.DataFrame({
        'True Label': trueL,
        'Predicted Label': preL
    })
    df.to_excel('/mnt/bc8f2e4d-b1c3-4772-8cbb-68d7a51e2523/xpg/TrainFramework/prediction_results.xlsx', index=False)
    metrics = metrics_calculator.compute_all(predictions, ground_truth)

    print("\n=== Test Metrics ===")
    for task_type, metric_dict in metrics.items():
        print(f"[{task_type}]")
        for name, value in metric_dict.items():
            print(f"  {name}: {value:.4f}")

if __name__ == "__main__":
    test()
