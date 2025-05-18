import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
from src.core.trainer import GenericTrainer
from hydra import utils

import os
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
import mlflow

@hydra.main(config_path="configs", config_name="exp4", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # 如果开启DDP，则初始化分布式
    ddp_enabled = cfg.training.parameter.get("ddp", True)
    if ddp_enabled:
        dist.init_process_group(
            backend="nccl",      # 通常在GPU环境下使用 nccl
            init_method="env://"
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0  # 单机单卡或者非分布式情况下，默认为0

    # 如果只想在 rank=0 上输出，可以判断：
    is_main_process = (local_rank == 0)

    # 让Hydra把内部的变量解析掉
    OmegaConf.resolve(cfg)
    
    # 只有在主进程上才去初始化MLflow实验
    if is_main_process:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.experiment.name)

    if is_main_process:
        # 记录一些通用参数
        with mlflow.start_run(run_name=cfg.experiment.run_name):
            mlflow.log_params({
                "model": cfg.model._target_,
                "optimizer": cfg.optimizer._target_,
                "scheduler": cfg.scheduler._target_,
                "batch_size": cfg.training.parameter.batch_size,
                "epochs": cfg.training.parameter.epochs,
            })
            
            # 创建训练器并开始训练
            trainer = GenericTrainer(cfg, local_rank=local_rank, is_main_process=is_main_process, ddp_enabled=ddp_enabled)
            trainer.run()

    else:
        # 对于其他进程，不需要启动新的 mlflow.run()
        # 直接创建Trainer并训练
        trainer = GenericTrainer(cfg, local_rank=local_rank, is_main_process=is_main_process, ddp_enabled=ddp_enabled)
        trainer.run()

    # 训练结束后销毁进程组
    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
    #启动命令 torchrun --nproc_per_node=2 train.py