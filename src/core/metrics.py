# core/metrics.py
import torch
import numpy as np
from torch import nn
from typing import Optional, Dict, List
from sklearn.metrics import f1_score

class RegressionMetrics(nn.Module):
    """回归任务评估指标集合"""
    def __init__(self, 
                 metrics: list = ['mae', 'mse', 'r2'], 
                 epsilon: float = 1e-8,
                 log_metrics: bool = False):
        """
        Args:
            metrics: 需要计算的指标列表，支持：
                'mae' - 平均绝对误差
                'mse' - 均方误差
                'rmse' - 均方根误差
                'r2' - 决定系数
                'mape' - 平均绝对百分比误差
                'msle' - 均方对数误差
            epsilon: 防止除零的小量
            log_metrics: 是否计算对数空间指标
        """
        super().__init__()
        self.epsilon = epsilon
        self.log_metrics = log_metrics
        self.metric_fns = self._build_metrics(metrics)
        
    def _build_metrics(self, metric_names):
        """构造指标计算函数字典"""
        registry = {
            'mae': self.mean_absolute_error,
            'mse': self.mean_squared_error,
            'rmse': self.root_mean_squared_error,
            'r2': self.r_squared,
            'mape': self.mean_absolute_percentage_error,
            'msle': self.mean_squared_log_error
        }
        return {name: registry[name] for name in metric_names if name in registry}

    def forward(self, 
               y_pred: torch.Tensor, 
               y_true: torch.Tensor,
              ) -> Dict[str, float]:
        """
        Args:
            y_pred: 预测值张量 (N, ...)
            y_true: 真实值张量 (N, ...)
        Returns:
            包含各指标值的字典
        """
        if self.log_metrics:
            y_pred = torch.log(y_pred + self.epsilon)
            y_true = torch.log(y_true + self.epsilon)
            
        results = {}
        for name, fn in self.metric_fns.items():
            results[name] = fn(y_pred, y_true).item()
        return results

    @staticmethod
    def _safe_divide(numerator: torch.Tensor, 
                    denominator: torch.Tensor,
                    epsilon: float = 1e-8) -> torch.Tensor:
        """安全的除法操作"""
        return numerator / (denominator + epsilon)
    
    def mean_absolute_error(self, 
                           y_pred: torch.Tensor, 
                           y_true: torch.Tensor,
                           ) -> torch.Tensor:
        """MAE = mean(|y_true - y_pred|)"""
        error = torch.abs(y_pred - y_true)
        return torch.mean(error)

    def mean_squared_error(self,
                          y_pred: torch.Tensor,
                          y_true: torch.Tensor,
                          ) -> torch.Tensor:
        """MSE = mean((y_true - y_pred)^2)"""
        error = (y_pred - y_true)**2
        return torch.mean(error)

    def root_mean_squared_error(self,
                               y_pred: torch.Tensor,
                               y_true: torch.Tensor,
                               ) -> torch.Tensor:
        """RMSE = sqrt(MSE)"""
        return torch.sqrt(self.mean_squared_error(y_pred, y_true))

    def r_squared(self,
                 y_pred: torch.Tensor,
                 y_true: torch.Tensor,
                 ) -> torch.Tensor:
        """R² = 1 - SS_res / SS_tot"""
        ss_res = torch.sum((y_true - y_pred)**2)
        ss_tot = torch.sum((y_true - torch.mean(y_true))**2)
        #print(f"SS_res: {ss_res.item()}, SS_tot: {ss_tot.item()}")

        return 1 - self._safe_divide(ss_res, ss_tot)

    def mean_absolute_percentage_error(self,
                                      y_pred: torch.Tensor,
                                      y_true: torch.Tensor,
                                      ) -> torch.Tensor:
        """MAPE = mean(|(y_true - y_pred)/y_true|)"""
        relative_error = torch.abs((y_true - y_pred) / (y_true + self.epsilon))
        return 100 * torch.mean(relative_error)

    def mean_squared_log_error(self,
                              y_pred: torch.Tensor,
                              y_true: torch.Tensor,
                              ) -> torch.Tensor:
        """MSLE = mean((log(y_true + 1) - log(y_pred + 1))^2)"""
        log_error = torch.log(y_true + 1) - torch.log(y_pred + 1)
        squared_log_error = log_error**2
        return torch.mean(squared_log_error)

class ClassificationMetrics(nn.Module):
    """分类任务评估指标集合"""
    def __init__(self, metrics: List[str] = ['accuracy', 'f1']):
        super().__init__()
        self.metric_fns = self._build_metrics(metrics)
    
    def _build_metrics(self, metric_names: List[str]) -> Dict[str, callable]:
        registry = {
            'accuracy': self.accuracy,
            'f1': self.f1_score_macro
        }
        return {name: registry[name] for name in metric_names if name in registry}
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        return {name: fn(y_pred, y_true) for name, fn in self.metric_fns.items()}
    
    @staticmethod
    def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        pred_labels = torch.argmax(y_pred, dim=1).unsqueeze(-1)
        return (pred_labels == y_true).float().mean().item()
    @staticmethod
    def f1_score_macro(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
        true_labels = y_true.cpu().numpy()
        return f1_score(true_labels, pred_labels, average='macro')

class MetricRegistry:
    """指标注册器，用于管理所有可用的评估指标"""
    def __init__(self):
        self._metrics = {}
        self._task_types = {}
    
    def register(self, name: str, metric_fn: callable, task_type: str = "all"):
        """注册新的评估指标
        Args:
            name: 指标名称
            metric_fn: 指标计算函数
            task_type: 指标适用的任务类型，可以是 'regression', 'classification' 或 'all'
        """
        self._metrics[name] = metric_fn
        self._task_types[name] = task_type
        
    def get_metric(self, name: str) -> callable:
        """获取指标计算函数"""
        if name not in self._metrics:
            raise ValueError(f"Unknown metric: {name}")
        return self._metrics[name]
    
    def get_task_type(self, name: str) -> str:
        """获取指标适用的任务类型"""
        return self._task_types.get(name, "all")
    
    def list_metrics(self, task_type: Optional[str] = None) -> List[str]:
        """列出所有可用的指标
        Args:
            task_type: 如果指定，只返回适用于该任务类型的指标
        """
        if task_type is None:
            return list(self._metrics.keys())
        return [name for name, t in self._task_types.items() 
                if t == task_type or t == "all"]

class MetricsCalculator:
    """指标计算器，支持混合任务的指标计算"""
    def __init__(self, 
                 metrics_config: Dict[str, List[str]],
                 registry: Optional[MetricRegistry] = None):
        """
        Args:
            metrics_config: 指标配置字典，格式为:
                {
                    "regression": ["mae", "mse", ...],
                    "classification": ["accuracy", "f1", ...],
                }
            registry: 指标注册器实例，如果为None则使用默认注册器
        """
        self.registry = registry or self._create_default_registry()
        self.metrics = {}
        
        # 为每种任务类型初始化指标
        for task_type, metric_names in metrics_config.items():
            self.metrics[task_type] = []
            for name in metric_names:
                metric_task = self.registry.get_task_type(name)
                if metric_task != "all" and metric_task != task_type:
                    raise ValueError(f"Metric {name} not supported for task {task_type}")
                self.metrics[task_type].append((name, self.registry.get_metric(name)))
    
    @staticmethod
    def _create_default_registry():
        """创建并配置默认的指标注册器"""
        registry = MetricRegistry()
        
        # 注册回归指标
        regression_metrics = RegressionMetrics()
        for name, fn in regression_metrics._build_metrics(['mae', 'mse', 'rmse', 'r2', 'mape', 'msle']).items():
            registry.register(name, fn, "regression")
        # 注册分类指标
        classification_metrics=ClassificationMetrics()
        for name, fn in classification_metrics._build_metrics(["accuracy", "f1"]).items():
            registry.register(name, fn, "classification")
        return registry
    
    def compute(self, 
                task_type: str,
                predictions: torch.Tensor, 
                targets: torch.Tensor,
               ) -> Dict[str, float]:
        """计算指定任务类型的所有指标
        Args:
            task_type: 任务类型 ('regression' 或 'classification')
            predictions: 模型预测值
            targets: 真实标签
        Returns:
            包含所有指标计算结果的字典
        """
        if task_type not in self.metrics:
            raise ValueError(f"Unknown task type: {task_type}")
            
        results = {}
        for name, metric_fn in self.metrics[task_type]:
            try:
                # 确保输入张量不为空
                if predictions is None or targets is None:
                    results[name] = 0.0
                    continue
                    
                # 转换为 CPU 张量并计算指标
                pred = predictions.detach().cpu()
                tgt = targets.detach().cpu()
                
                if name == 'f1':
                    # 对于分类任务，转换为类别索引
                    score = metric_fn(pred, tgt)
                else:
                    score = metric_fn(pred, tgt)
                
                results[name] = score.item() if torch.is_tensor(score) else float(score)
            except Exception as e:
                print(f"计算 {name} 指标时出错: {str(e)}")
                results[name] = 0.0
        return results

    def compute_all(self,
                   predictions: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor],
                   ) -> Dict[str, Dict[str, float]]:
        """计算所有任务类型的指标
        Args:
            predictions: 各任务的预测值字典
            targets: 各任务的真实标签字典
        Returns:
            嵌套字典，包含所有任务类型的指标结果
        """
        results = {}
        for task_type in self.metrics.keys():
            if task_type in predictions and task_type in targets:
                results[task_type] = self.compute(
                    task_type,
                    predictions[task_type],
                    targets[task_type],
                )
        return results

if __name__ == "__main__":
    # 单元测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试数据
    # y_true = torch.tensor([3, -0.5, 2, 7], dtype=torch.float32, device=device)
    # y_pred = torch.tensor([2.5, 0.0, 2, 8], dtype=torch.float32, device=device)
    y_true = torch.tensor([0, 1, 2, 2,1,0,2,2,1,1,0,0], dtype=torch.float32, device=device)
    y_pred = torch.tensor([[0.5,0.3,0.2], [0.2,0.5,0.2], [0.5,0.3,0.7], [0.5,0.3,0.2],[0.2,0.5,0.2],[0.5,0.3,0.2],[0.5,0.3,0.7],[0.2,0.5,0.2],[0.2,0.5,0.2],[0.2,0.5,0.2],[0.5,0.3,0.2],[0.5,0.3,0.2]], dtype=torch.float32, device=device)
    
    # 计算指标
    metrics = RegressionMetrics(metrics=['mae', 'mse', 'r2', 'mape'])
    metrics_class = ClassificationMetrics(metrics=['accuracy', 'f1'])
    #results = metrics(y_pred, y_true)
    results=metrics_class(y_pred, y_true)
    
    # 预期结果（基于sklearn计算）
    # expected = {
    #     'mae': 0.5,         # (0.5 + 0.5 + 0 + 1)/4 = 0.5
    #     'mse': 0.375,       # (0.5² + 0.5² + 0 + 1²)/4 = 0.375
    #     'r2': 0.948,     # 1 - (1.5 / 29.0)
    #     'mape': 32.738   # 平均百分比误差
    # }
    expected = {
        'accuracy': 0.5,         # (0.5 + 0.5 + 0 + 1)/4 = 0.5
        'f1': 0.375,       # (0.5² + 0.5² + 0 + 1²)/4 = 0.375
    }

    
    print("测试结果:")
    for k, v in results.items():
        print(f"{k}: {v:.4f} (预期: {expected.get(k, 'N/A'):.4f})")