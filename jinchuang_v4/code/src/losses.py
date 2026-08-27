"""
分层双Margin Triplet Loss

设计思路:
  m1（普通间隔）: 同客户 vs 普通不同人，保证基础人脸区分
  m2（欺诈间隔）: 同客户 vs 跨客户高仿，额外加大隔离区间

对每个三元组，动态计算 anchor-negative 的余弦相似度:
  - 相似度 > threshold → 判为"高仿负样本"，用 m2 大间隔拉开
  - 相似度 ≤ threshold → 普通负样本，用 m1 正常拉开

这样在特征空间中，欺诈样本（翻拍、AI换脸、意外高相似）会被推得更远，
更容易在阈值判定时触发预警。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualMarginTripletLoss(nn.Module):
    """分层双Margin Triplet Loss

    Args:
        m1: 普通间隔，同客户 vs 普通陌生人
        m2: 欺诈间隔，同客户 vs 跨客户高仿（m2 > m1）
        sim_threshold: 余弦相似度阈值，超过此值判为高仿负样本
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, m1: float = 0.3, m2: float = 0.8,
                 sim_threshold: float = 0.85, reduction: str = "mean"):
        super().__init__()
        assert m2 > m1, "m2（欺诈间隔）必须大于 m1（普通间隔）"
        self.m1 = m1
        self.m2 = m2
        self.sim_threshold = sim_threshold
        self.reduction = reduction

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """计算分层双Margin Triplet Loss

        Args:
            anchor: [batch, D] 锚点特征（L2归一化后）
            positive: [batch, D] 同类正样本特征
            negative: [batch, D] 异类负样本特征

        Returns:
            标量 loss
        """
        # L2 距离：在归一化空间中等价于余弦距离的单调变换
        d_ap = torch.norm(anchor - positive, p=2, dim=-1)  # [batch]
        d_an = torch.norm(anchor - negative, p=2, dim=-1)  # [batch]

        # 动态判断：每个负样本是普通还是高仿
        with torch.no_grad():
            cos_sim = F.cosine_similarity(anchor, negative, dim=-1)  # [batch]
            margins = torch.where(
                cos_sim > self.sim_threshold,
                torch.full_like(cos_sim, self.m2),
                torch.full_like(cos_sim, self.m1),
            )

        # Triplet loss: max(0, d_ap - d_an + margin)
        losses = F.relu(d_ap - d_an + margins)

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        return losses

    def extra_repr(self) -> str:
        return f"m1={self.m1}, m2={self.m2}, sim_threshold={self.sim_threshold}"


class DualMarginTripletLossWithStats(DualMarginTripletLoss):
    """带统计追踪的 DualMarginTripletLoss，用于训练时日志输出"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_stats()

    def reset_stats(self):
        self.high_sim_count = 0
        self.low_sim_count = 0
        self.batch_count = 0

    def forward(self, anchor, positive, negative):
        with torch.no_grad():
            cos_sim = F.cosine_similarity(anchor, negative, dim=-1)
            self.high_sim_count += (cos_sim > self.sim_threshold).sum().item()
            self.low_sim_count += (cos_sim <= self.sim_threshold).sum().item()
            self.batch_count += 1

        return super().forward(anchor, positive, negative)

    def get_ratio(self):
        total = self.high_sim_count + self.low_sim_count
        if total == 0:
            return 0.0
        return self.high_sim_count / total
