import torch
import torch.nn as nn
from ..blocks.conv_blocks import Conv,ConvNextBlock
import torch.nn.functional as F

class Posture_network(nn.Module):
    def __init__(self,n_chans=768,n_blocks=1):
        super().__init__()
        self.n_chans=n_chans
        self.conv1=nn.Conv2d(n_chans,n_chans,kernel_size=3,padding=1)
        self.blocks=nn.Sequential(*(n_blocks*[ConvNextBlock(dim=n_chans)]))

    def forward(self,x):
        out=torch.relu(self.conv1(x))
        out=self.blocks(out)
        return out

# 单纯的体重回归头
class SWRHead(nn.Module):
    def __init__(self, ch=128, nc=3):
        super().__init__()
        self.nc = nc  # number of classes
        self.merge = Conv(ch, 768, 1, 1,act=nn.GELU())

        # 回归分支
        self.reg = nn.Sequential(
                    Conv(768, 768, 3, 1, 1, act=nn.ReLU()),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.LayerNorm(768),
                    nn.Linear(768, 1)
        )


    def forward(self, x):
        x = self.merge(x)
        reg_out = self.reg(x)  # [B, 1]
        # reg_out = torch.sigmoid(reg_out)

        return torch.cat([reg_out], 1)  # 把分类和回归结果按channel维度，即dim=1拼接

#分开分类和体重回归，可以更快收敛
class DecoupleHead(nn.Module):
    def __init__(self, ch=128, nc=3):
        super().__init__()
        self.nc = nc  # number of classes
        self.merge = Conv(ch, 768, 1, 1,act=nn.GELU())

        # 分类分支
        self.cls = nn.Sequential(
            Conv(768, 768, 3, 1, 1, act=nn.ReLU()),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Linear(768, 3)
        )

        # 回归分支
        self.reg = nn.Sequential(
                    Conv(768, 768, 3, 1, 1, act=nn.ReLU()),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.LayerNorm(768),
                    nn.Linear(768, 1)
        )


    def forward(self, x):
        x = self.merge(x)

        cls_out = self.cls(x)  # [B, nc]
        reg_out = self.reg(x)  # [B, 1]
        # reg_out = torch.sigmoid(reg_out)

        return torch.cat([cls_out,reg_out], 1)  # 把分类和回归结果按channel维度，即dim=1拼接


#结合分类和体重回归，理论是姿态会影响体重预测，但是训练会更难
class CoupleHead(nn.Module):
    def __init__(self, _,ch=1344,nc=3):
        super().__init__()
        self.nc = nc
        self.flatten = nn.Flatten()
        # 特征提取
        self.feature_extractor = Posture_network(ch)
        # 分类分支
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(ch, eps=1e-6),
            nn.GELU(),
            nn.Linear(ch, nc)
        )

        # 回归分支（使用 features + original_input）
        self.reg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(ch, eps=1e-6),
            nn.GELU(),
            nn.Linear(ch, 1)
        )

    def forward(self, x):
        original_input = x
        features = self.feature_extractor(x)

        # 分类分支
        cls_out = self.cls(features)

        # 回归分支
        reg_input = features + original_input
        reg_out = self.reg(reg_input)
        # reg_out = torch.sigmoid(reg_out)

        return torch.cat([cls_out,reg_out], 1)



class MY_Weight_Regression_Head(nn.Module):
    """Adaptive regression head that automatically handles input dimensions"""
    
    def __init__(self, ch: int, hidden_dim: int = 256):
        """
        Initialize the detection head
        
        Args:
            ch: Number of input channels
            hidden_dim: Hidden layer dimension (default: 256)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Defer creation of layers until we see the input shape
        self.flatten = nn.Flatten(1, -1)
        self.initialized = False
        self.linear1 = None
        self.linear2 = None
        self.act = nn.GELU()

    def _initialize_layers(self, input_dim: int):
        """
        Initialize the linear layers based on input dimension
        
        Args:
            input_dim: Flattened input dimension
        """
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cuda'
        self.linear1 = nn.Linear(input_dim, self.hidden_dim).to(device)
        self.linear2 = nn.Linear(self.hidden_dim, 1).to(device)
        self.initialized = True
        
        # Log the architecture
        print(f"Initialized regression head with input dim: {input_dim}, "
              f"hidden dim: {self.hidden_dim}, device: {device}")

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor or list of tensors. If list, uses first tensor.
        
        Returns:
            Regression output
        """
        # Handle input if it's a list
        if isinstance(x, (list, tuple)):
            x = x[0]
        
        # Get the device of input tensor
        device = x.device
            
        # Flatten the input
        x = self.flatten(x)
        
        # Initialize layers on first forward pass if not done
        if not self.initialized:
            input_dim = x.shape[1]  # Get flattened dimension
            self._initialize_layers(input_dim)
            
        # Ensure all components are on the same device
        if not self.initialized:
            return torch.zeros(x.shape[0], 1, device=device)
            
        # Move layers to the correct device if needed
        if self.linear1.weight.device != device:
            self.linear1 = self.linear1.to(device)
            self.linear2 = self.linear2.to(device)
            
        # Forward pass through layers
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return x

    def __repr__(self):
        """String representation of the module"""
        return (f'{self.__class__.__name__}('
                f'hidden_dim={self.hidden_dim}, '
                f'initialized={self.initialized})')