import torch

class DataNormalizer:
    def __init__(self, y_true):
        """
        初始化时计算均值和标准差
        """
        self.mean = torch.tensor(y_true).mean()
        self.std = torch.tensor(y_true).std()

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        """
        标准化：将 y 转换为 (y - mean) / std
        """
        return (y - self.mean) / self.std

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        """
        反标准化：将 y 还原回原始尺度
        """
        return y * self.std + self.mean
