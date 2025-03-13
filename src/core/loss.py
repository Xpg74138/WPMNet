import torch
import torch.nn as nn
from torchmetrics import LogCoshError

class PWLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化MSE损失函数（不需要参数）
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        """
        前向计算
        参数:
            predictions: 模型预测输出 (shape: [batch, n_outputs])
            targets: 真实标签 (shape: [batch, n_targets])
        返回:
            loss: 总损失值
        """
        # 自动获取当前设备信息（替代硬编码的'cuda'）
        predictions=list(predictions.values())
        device = predictions[0].device
        
        # 提取权重分量（使用targets的第一列）
        regression = targets["regression"].to(device)
        regression_pred=predictions[0]


        # 计算权重损失（使用predictions的第一列）
        lregression = self.mse(regression_pred, regression[:,0]).unsqueeze(0)

        total_loss=lregression

        return total_loss, torch.cat((
                lregression.detach().cpu(),
                total_loss.detach().cpu()
            ))
    
    def onehot(data):
        nums=data.shape[0]
        label_onehot=torch.zeros([nums,3])
        data=data.long().squeeze()
        label_onehot[torch.arange(nums),data]=1
        return torch.Tensor(label_onehot)

class PWAPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化MSE损失函数（不需要参数）
        self.regressLoss = nn.MSELoss().to(device)
        #self.regressLoss=LogCoshError().to(device)
        self.clsLoss = nn.CrossEntropyLoss().to(device)
        # 可以在此添加其他需要学习的参数
        # 例如：self.weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, predictions, targets):
        """
        前向计算
        参数:
            predictions: 模型预测输出 (shape: [batch, n_outputs])
            targets: 真实标签 (shape: [batch, n_targets])
        返回:
            loss: 总损失值
            loss_components: 各损失分量组成的张量
        """
        # 自动获取当前设备信息（替代硬编码的'cuda'）
        predictions=list(predictions.values())
        device = predictions[0].device
        
        # 提取权重分量（使用targets的第一列）
        regression = targets["regression"].to(device)
        classification=self.onehot(targets["classification"]).to(device)

        regression_pred=predictions[0]
        classification_pred=predictions[1]


        # 计算权重损失（使用predictions的第一列）
        lregression = self.regressLoss(regression_pred, regression).unsqueeze(0)
        
        # 计算分类损失
        lclassification = self.clsLoss(classification_pred, classification).unsqueeze(0)
        
        # 合并损失项
        total_loss = lclassification + 10*lregression
        
        # 返回总损失和各损失分量（自动转移到CPU）
        return total_loss, torch.cat((
            lclassification.detach().cpu(),
            lregression.detach().cpu(),
            total_loss.detach().cpu()
        ))
    
    def onehot(self,data):
        nums=data.shape[0]
        label_onehot=torch.zeros([nums,3])
        data=data.long().squeeze()
        label_onehot[torch.arange(nums),data]=1
        return torch.Tensor(label_onehot)