import matplotlib.pyplot as plt
import mlflow
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
from omegaconf import DictConfig,ListConfig
import thop
from hydra.utils import instantiate
from ..models import common_blocks,repeat_blocks,heads,multipleinput_blocks,func_blocks
import math
from colorama import Fore, Back, Style, init
from thop import profile
from ptflops import get_model_complexity_info
init(autoreset=True)

class ModelBuilder(nn.Module):
    """模块化模型构建器"""
    
    def __init__(self, cfg: DictConfig, nc: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.nc = nc if nc is not None else cfg.get("nc", 3)
        self.image_channel= cfg.input_channel if cfg.input_channel else 3
        self.channels = [] 
        self.layers = nn.ModuleList()
        self.save_indices = []
        self.module_registry = self._build_model_registry()

    def _build_model_registry(self) -> Dict[str, nn.Module]:
        """安全注册可用模块"""
        registry = {}
        # 合并所有模块字典
        for module_dict in [common_blocks, repeat_blocks, heads, multipleinput_blocks,func_blocks]:
            registry.update(module_dict)
        # 添加PyTorch内置模块
        for name in ['BatchNorm2d', 'Conv2d']:
            if hasattr(nn, name):
                registry[name] = getattr(nn, name)
        return registry
    
    def parse(self) -> Tuple[nn.Sequential, List[int]]:
        """解析配置并构建模型"""
        print(f"\n{'':>3}{'From':>18}{'N':>3}{'Params':>10}  {'Module':<40}{'Arguments':<30}")
        backbone_head = self.cfg["Backbone"] + self.cfg["Head"]
        
        for i, (f, n, m, args) in enumerate(backbone_head):
            if not isinstance(f,int):
                f=list(f)
            module, c2 = self._parse_layer(i, f, n, m, args)
            self.layers.append(module)
            self.channels.append(c2)
            
        return nn.Sequential(*self.layers), sorted(self.save_indices)

    def _parse_layer(self, index: int, f: Union[int, List], n: int, m: str, args: list):
        """解析单层配置"""
        # 参数预处理
        module = self._safe_get_module(m)
        args = self._parse_args(args)
        original_n = n  # 保存原始n值
        module_type = module.__name__  # 获取原始模块类型名称
        
        # 处理输入通道
        if isinstance(f, list):
            c1 = sum(self.channels[x] for x in f)
        else:
            if len(self.channels):
                c1 =self.channels[f] 
            else:
                c1=self.image_channel 
        
        c2 = None  # 初始化输出通道
        
        # 特殊模块处理逻辑
        if module in multipleinput_blocks.values():
            c2 = sum(self.channels[x] for x in f)
        elif module is nn.BatchNorm2d:
            # BN层参数处理
            c2 = self.channels[f]
            args = [c2]
            
        elif module in heads.values():
            # 检测头添加输入通道列表
            args.append(self.channels[f])
            c2 = self.channels[f]  # 假设输出通道不变
            
        elif module in common_blocks.values():
            # 通用模块处理（如Conv）
            c2 = args[0]
            args = [c1, c2, *args[1:]]

            # 处理重复块（如C3）
            if module in repeat_blocks.values():
                args.insert(2, original_n)  # 插入重复次数到参数
                n = 1  # 重置层数
        else:
            # 默认情况
            c2 = self.channels[f]
        
        # 构建模块
        module_seq = self._build_module(module, n, args)
        
        # 添加元数据属性
        module_seq.i = index           # 层索引
        module_seq.f = f               # 输入来源
        module_seq.type = module_type  # 原始模块类型
        module_seq.np = sum(p.numel() for p in module_seq.parameters())  # 参数总数
        module_seq.args= args

        # 记录需要保存的层
        if isinstance(f, list):
            self.save_indices.extend([x for x in f if x != -1])
        elif f != -1:
            self.save_indices.extend([f])

        # 打印信息
        self._print_layer_info(module_seq)
        return module_seq, c2

    def _safe_get_module(self, name: str) -> nn.Module:
        """安全获取模块"""
        if name in self.module_registry:
            return self.module_registry[name]
        raise ValueError(f"未注册的模块类型: {name}")

    def _parse_args(self, args: list) -> list:
        """参数安全求值"""
        parsed = []
        for a in args:
            if isinstance(a, str):
                try:
                    parsed.append(eval(a))
                except (NameError, SyntaxError):
                    parsed.append(a)
            else:
                parsed.append(a)
        return parsed

    def _build_module(self, module: nn.Module, n: int, args: list) -> nn.Module:
        """构建重复模块"""
        if n > 1:
            container = nn.Sequential(*[module(*args) for _ in range(n)])
            container.type = module.__class__.__name__  # 覆盖Sequential类型
            return container
        return module(*args)

    def _print_layer_info(self, module: nn.Module):
        """格式化打印层信息"""
        print(Fore.GREEN+f"{module.i:>3}{str(module.f):>18}{getattr(module, 'n', 1):>3}"
              f"{module.np:10.0f}  {module.type:<40}{str(module.args):<30}")

class XModel(nn.Module):
    """支持多任务的通用模型"""
    
    def __init__(self, 
                 cfg: DictConfig,
                 task: str = 'detection',
                 visualize: bool = False):
        super().__init__()
        self.task = task
        self.visualize = visualize
        self.builder = ModelBuilder(cfg)
        self.model ,self.save= self.builder.parse()
        self._init_weights(cfg.get('weights'))
        self.profile= False
        
    def _init_weights(self, weights_path: Optional[str] = None):
        """权重初始化"""
        if weights_path:
            print(Fore.GREEN+f"Loading model weights from path {weights_path}")
            self.load_weights(weights_path)
        else:
            print(Fore.GREEN+"Initializing model weights...")
            self.apply(self._kaiming_init)
            # 添加线性层初始化更合理
            self.apply(self._init_linear_layers)
            
    def _kaiming_init(self, m: nn.Module):
        """改进的初始化策略"""
        if isinstance(m, nn.Conv2d):
            # 自动计算fan_in/fan_out（对深度可分离卷积更友好）
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            if fan_out != 0:  # 防止除零错误
                gain = nn.init.calculate_gain('leaky_relu' if hasattr(m, 'act') and m.act is not None else 'relu', 0.1)
                std = gain / math.sqrt(fan_out)
                nn.init.normal_(m.weight, 0, std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
            # 初始化running stats
            nn.init.constant_(m.running_mean, 0)
            nn.init.constant_(m.running_var, 1)
            
    def _init_linear_layers(self, m: nn.Module):
        """单独处理全连接层"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 处理LayerNorm层
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    
    def load_weights(self, weights_path: str):
        """加载预训练权重"""
        try:
            checkpoint = torch.load(weights_path)
            state_dict = checkpoint.get('model_state', checkpoint)
            self.load_state_dict(state_dict, strict=True)
            mlflow.log_artifact(weights_path)  # 记录权重文件
        except Exception as e:
            print(Back.RED+f"权重加载失败: {str(e)}")
            print("使用随机初始化...")
            
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """多任务前向传播"""
        y=[] #各层输出存储
        
        for module in self.model:
            # 动态获取输入源
            x = self._get_input(module, y, x)
            
            # 执行前向计算
            x = module(x)
            
            # 保存需要缓存的输出
            y.append(x if module.i in self.save else None)
            
            # 性能分析（使用元数据）
            if self.profile:
                self._profile_layer(module, x)
        
        return self._format_output(x)
    
    def _get_input(self, module: nn.Module, y: list, default):
        """智能获取输入源"""
        if isinstance(module.f, list):  # 多输入情况
            return [y[i] if i != -1 else default for i in module.f]
        elif module.f != -1:            # 单输入
            return y[module.f]
        return default                  # 默认输入

    def _profile_layer(self, module: nn.Module, x: torch.Tensor):
        """带元数据的性能分析"""
        flops = thop.profile(module, inputs=(x,), verbose=False)[0] / 1e9 * 2
        print(Back.GREEN+f"{module.type} | FLOPs: {flops:.2f}G | Params: {module.np/1e6:.2f}M")

    def _format_output(self, raw_output: Dict) -> Dict:
        """根据任务格式化输出（新增）"""
        formatted = {}
        if self.task == 'weight':
            formatted['regression'] = raw_output[:, 0:1]
        elif self.task == 'weight_posture':
            formatted['regression'] = raw_output[:, 0:1]
            formatted['classification'] = raw_output[:, 1:]
        return formatted
    
    def log_info(self):
        # Model basic information
        total_params = sum(m.np for m in self.model)
        print(Fore.GREEN+f"\n{'-'*60}")
        print(Fore.GREEN+f"{'Model Info':^60}")
        print(Fore.GREEN+f"{'-'*60}")
        print(Fore.GREEN+f"Total params: {total_params/1e6:.2f}M")
        
        # Device and Precision of calculation
        print(Fore.GREEN+f"\n{'Device Info':^60}")
        print(Fore.GREEN+f"{'-'*60}")
        print(Fore.GREEN+f"Device: {next(self.parameters()).device}")
        print(Fore.GREEN+f"Precision: {next(self.parameters()).dtype}")
        
        print(Fore.GREEN+f"{'-'*60}")