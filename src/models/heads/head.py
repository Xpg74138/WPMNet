import torch
import torch.nn as nn
from ..blocks.conv_blocks import Conv,ConvNextBlock


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

#分开分类和体重回归，可以更快收敛
class DecoupleHead(nn.Module):
    def __init__(self, ch=256, nc=3):
        super().__init__()
        self.nc = nc  # number of classes
        self.merge = Conv(ch, 768, 1, 1,act=nn.GELU())
        self.cls_convs1 = Conv(768, 768, 3, 1, 1,act=nn.GELU())
        self.weight_convs1 = Conv(768, 768, 3, 1, 1,act=nn.GELU())
        self.cls_norm=nn.LayerNorm(768, eps=1e-6)
        self.weight_norm=nn.LayerNorm(768, eps=1e-6)
        self.cls_preds = nn.Linear(768, 3, 1)
        self.weight_preds = nn.Linear(768, 1 , 1)

    def forward(self, x):
        x = self.merge(x)
        x1 = self.cls_convs1(x)
        x1=self.cls_norm(x1.mean([-2,-1]))
        x1 = self.cls_preds(x1)

        x2 = self.weight_convs1(x)
        x2=self.weight_norm(x2.mean([-2,-1]))
        x2 = self.weight_preds(x2)
        out = torch.cat([x1,x2], 1)  # 把分类和回归结果按channel维度，即dim=1拼接
        return out
    
#结合分类和体重回归，理论是姿态会影响体重预测，但是训练会更难
class CoupleHead(nn.Module):
    def __init__(self, ch,nc=3):
        super().__init__()
        self.nc = nc
        # 姿态分类
        self.head = nn.Linear(1344, 3)
        self.flatten = nn.Flatten()
        # 体重回归
        self.fc1 = nn.Linear(1344, 1)
        self.act = nn.GELU()
        self.mynorm1 = nn.LayerNorm(1344, eps=1e-6)
        self.mynorm2 = nn.LayerNorm(1344, eps=1e-6)
        self.addednetwork = Posture_network(1344)

    def forward(self, x):
        x=x
        x_=x
        # 获取姿态分类网络的输出和特征
        mid = self.addednetwork(x)
        x = self.mynorm1(mid.mean([-2, -1]))
        x = self.head(x)
        # 体重回归网络
        y = mid + x_
        y = self.mynorm2(y.mean([-2, -1]))
        y = self.fc1(y)
        return torch.cat([x,y], 1)



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