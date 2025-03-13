import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
from src.core.trainer import GenericTrainer
from hydra import utils

@hydra.main(config_path="configs", config_name="exp2", version_base="1.3")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    # 设置MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)
    
    # 记录参数
    with mlflow.start_run(run_name=cfg.experiment.run_name):
        mlflow.log_params({
            "model": cfg.model._target_,
            "optimizer": cfg.optimizer._target_,
            "scheduler": cfg.scheduler._target_,
            "batch_size": cfg.training.parameter.batch_size,
            "epochs": cfg.training.parameter.epochs,
        })
        
        # 创建训练器并开始训练
        trainer = GenericTrainer(cfg)
        trainer.run()

if __name__ == "__main__":
        main()