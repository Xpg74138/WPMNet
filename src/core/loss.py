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
        total_loss = lclassification + lregression
        
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

class PWAPLoss_weighting(nn.Module):
    #这个改动的核心思路来自：
    # "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (Kendall et al., CVPR 2018)
    # 它的思想是：
    # 如果一个任务的不确定性大（即噪声大），那它的损失对整体训练的影响应该小；反之亦然。
    # 通过对 log_var 的优化，模型能自动调整每个任务损失的相对比例，实现更好的平衡。
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.regressLoss = nn.MSELoss().to(device)
        self.clsLoss = nn.CrossEntropyLoss().to(device)

        # 添加可学习的 log(σ^2)，初始设为0
        self.log_var_reg = nn.Parameter(torch.tensor(0.0))  # 回归损失的 log-variance
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))  # 分类损失的 log-variance

    def forward(self, predictions, targets):
        predictions = list(predictions.values())
        device = predictions[0].device
        
        regression = targets["regression"].to(device)
        classification = self.onehot(targets["classification"]).to(device)

        regression_pred = predictions[0]
        classification_pred = predictions[1]

        lreg = self.regressLoss(regression_pred, regression)
        lcls = self.clsLoss(classification_pred, classification)

        # 使用不确定性加权多任务损失
        loss = (
            torch.exp(-self.log_var_reg) * lreg + self.log_var_reg + 
            torch.exp(-self.log_var_cls) * lcls + self.log_var_cls
        )

        return loss, torch.cat((
            lcls.detach().cpu().unsqueeze(0),
            lreg.detach().cpu().unsqueeze(0),
            loss.detach().cpu().unsqueeze(0)
        ))
    
    def onehot(self, data):
        nums = data.shape[0]
        label_onehot = torch.zeros([nums, 3], device=data.device)
        data = data.long().squeeze()
        label_onehot[torch.arange(nums), data] = 1
        return label_onehot

class PWAPLogMSELoss(PWAPLoss):
    """
    对数变换 + MSE 损失函数
    适用场景：多尺度体重回归（30kg-150kg），通过对数压缩尺度差异
    特点：
        1. 对体重标签取自然对数后做MSE，强调相对误差
        2. 输出时自动用exp()还原为线性值
    """
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-6  # 避免log(0)

    def forward(self, predictions, targets):
        predictions = list(predictions.values())
        device = predictions[0].device
        
        # 对数变换标签
        regression = torch.log(targets["regression"].to(device) + self.epsilon)
        classification = self.onehot(targets["classification"]).to(device)
        
        # 模型输出直接作为log(weight)
        regression_pred = predictions[0]
        
        # 计算对数空间MSE
        lregression = self.regressLoss(regression_pred, regression).unsqueeze(0)
        lclassification = self.clsLoss(predictions[1], classification).unsqueeze(0)
        
        total_loss = lclassification + 10 * lregression
        
        return total_loss, torch.cat((
            lclassification.detach().cpu(),
            lregression.detach().cpu(),
            total_loss.detach().cpu()
        ))
    
    def inverse_transform(self, pred_log):
        """将模型输出的log(weight)还原为体重"""
        return torch.exp(pred_log)
    
class PWAPSILogLoss(PWAPLoss):
    """
    SILog损失函数（尺度不变对数误差）
    适用场景：需要严格保证相对误差一致性的体重回归
    特点：
        1. 直接优化log空间相对误差，适合多尺度数据
        2. 对30kg和150kg的10%误差给予相同惩罚
    """
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-6

    def silog_loss(self, pred, target):
        log_diff = torch.log(pred + self.epsilon) - torch.log(target + self.epsilon)
        return torch.mean(log_diff ** 2) - 0.5 * torch.mean(log_diff) ** 2

    def forward(self, predictions, targets):
        predictions = list(predictions.values())
        device = predictions[0].device
        
        regression = targets["regression"].to(device)
        classification = self.onehot(targets["classification"]).to(device)
        
        # 使用SILog代替MSE
        lregression = self.silog_loss(predictions[0], regression).unsqueeze(0)
        lclassification = self.clsLoss(predictions[1], classification).unsqueeze(0)
        
        total_loss = lclassification + 10 * lregression
        
        return total_loss, torch.cat((
            lclassification.detach().cpu(),
            lregression.detach().cpu(),
            total_loss.detach().cpu()
        ))
    
class PWAPMixedLoss(PWAPLoss):
    """
    混合损失函数（MSE + SILog）
    适用场景：需要平衡绝对误差和相对误差的体重回归
    参数：
        alpha: 控制MSE和SILog的权重（默认0.5）
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.epsilon = 1e-6

    def silog_loss(self, pred, target):
        log_diff = torch.log(pred + self.epsilon) - torch.log(target + self.epsilon)
        return torch.mean(log_diff ** 2) - 0.5 * torch.mean(log_diff) ** 2

    def forward(self, predictions, targets):
        predictions = list(predictions.values())
        device = predictions[0].device
        
        regression = targets["regression"].to(device)
        classification = self.onehot(targets["classification"]).to(device)
        
        # 计算混合损失
        mse_loss = self.regressLoss(predictions[0], regression)
        silog_loss = self.silog_loss(predictions[0], regression)
        lregression = (self.alpha * mse_loss + (1 - self.alpha) * silog_loss).unsqueeze(0)
        
        lclassification = self.clsLoss(predictions[1], classification).unsqueeze(0)
        total_loss = lclassification + 10 * lregression
        
        return total_loss, torch.cat((
            lclassification.detach().cpu(),
            lregression.detach().cpu(),
            total_loss.detach().cpu()
        ))
    
class PWAPWeightedLoss(PWAPLoss):
    """
    分区间加权损失函数
    适用场景：体重数据分布不均匀（如小样本区间需要更高权重）
    参数：
        bins: 区间分割点（如[30, 50, 100, 150]）
        weights: 各区间损失权重（如[2.0, 1.0, 0.5]）
    """
    def __init__(self, bins=[30, 50, 100, 150], weights=[2.0, 1.0, 0.5]):
        super().__init__()
        self.bins = torch.tensor(bins)
        self.weights = torch.tensor(weights)

    def forward(self, predictions, targets):
        predictions = list(predictions.values())
        device = predictions[0].device
        
        regression = targets["regression"].to(device)
        classification = self.onehot(targets["classification"]).to(device)
        
        # 计算区间权重掩码
        bin_indices = torch.bucketize(regression, self.bins.to(device)) - 1
        loss_weights = self.weights.to(device)[bin_indices]
        
        # 加权MSE
        squared_error = (predictions[0] - regression) ** 2
        lregression = (squared_error * loss_weights).mean().unsqueeze(0)
        
        lclassification = self.clsLoss(predictions[1], classification).unsqueeze(0)
        total_loss = lclassification + 10 * lregression
        
        return total_loss, torch.cat((
            lclassification.detach().cpu(),
            lregression.detach().cpu(),
            total_loss.detach().cpu()
        ))