import os
import subprocess
from pathlib import Path

def run_experiment(exp_name: str):
    """运行单个实验配置（会自动继承并覆盖默认配置）"""
    # 直接使用配置文件路径覆盖
    cmd = f"python train.py --config-path=configs --config-name={exp_name}"
    print(f"\n开始运行实验: {exp_name}")
    
    process = subprocess.Popen(
        cmd.split(),
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
        if config_file.stem not in ["default", "base"]:
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
    for exp in experiments:
        run_experiment(exp)

if __name__ == "__main__":
    main()