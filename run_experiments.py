import os
import subprocess
from pathlib import Path

NPROC_PER_NODE = int(os.environ.get("NPROC_PER_NODE", "2"))
TRAIN_EPOCHS = os.environ.get("TRAIN_EPOCHS", "30").strip()
SCHEDULER_T_MAX = os.environ.get("SCHEDULER_T_MAX", "100").strip()


def hydra_overrides() -> list[str]:
    """Runtime overrides for shorter training without compressing the LR schedule."""
    overrides = []
    if TRAIN_EPOCHS:
        overrides.append(f"training.parameter.epochs={TRAIN_EPOCHS}")
    if SCHEDULER_T_MAX:
        overrides.append(f"scheduler.T_max={SCHEDULER_T_MAX}")
    return overrides


def run_experiment(exp_name: str):
    """运行单个实验配置（会自动继承并覆盖默认配置）"""
    cmd = [
        "torchrun",
        f"--nproc_per_node={NPROC_PER_NODE}",
        "train.py",
        "--config-path=configs",
        f"--config-name={exp_name}",
        *hydra_overrides(),
    ]
    print(f"\n开始运行实验: {exp_name}")
    print("命令:", " ".join(cmd))
    if os.environ.get("DRY_RUN", "0") == "1":
        return
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # 实时输出日志
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    if process.returncode != 0:
        print(f"实验失败: {exp_name}")
        print(process.stderr.read())
    else:
        print(f"实验完成: {exp_name}")

def get_experiment_configs():
    """自动扫描configs文件夹中的配置文件"""
    experiments = []
    experiment_dir = Path("configs")
    
    if not experiment_dir.exists():
        print(f"错误: 实验配置目录不存在: {experiment_dir}")
        return experiments
    
    # 获取所有yaml文件
    for config_file in experiment_dir.glob("*.yaml"):
        # 排除default.yaml和base.yaml
        if config_file.stem in ["exp32"]:
            experiments.append(config_file.stem)
    
    # 排序实验配置名
    experiments.sort()
    return experiments

def main():
    """按顺序运行所有实验"""
    # 实验配置名列表（不需要.yaml后缀）
    experiments = get_experiment_configs()
    if not experiments:
        print("未找到任何实验配置，请确保configs目录下包含有效的配置文件")
        return
    print(f"找到以下实验配置: {', '.join(experiments)}")
    print(f"训练轮数: {TRAIN_EPOCHS or 'config default'}")
    print(f"Scheduler T_max: {SCHEDULER_T_MAX or 'config default'}")
    for exp in experiments:
        run_experiment(exp)

if __name__ == "__main__":
    main()
