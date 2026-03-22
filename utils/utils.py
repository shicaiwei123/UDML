import torch
import torch.nn as nn
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class CenterContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, intra_weight=1.0, inter_weight=1.0):
        """
        混合损失：类内损失（样本到中心的距离）+ 类间损失（中心之间的距离）
        :param margin: 类间最小距离阈值
        :param intra_weight: 类内损失权重
        :param inter_weight: 类间损失权重
        """
        super().__init__()
        self.margin = margin
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight

    def forward(self, features, labels, centers, confidences=None):
        """
        计算损失
        :param features: 样本特征 (batch_size, feature_dim)
        :param labels: 样本标签 (batch_size,)
        :param centers: 所有类中心 (num_classes, feature_dim)
        :param confidences: 样本置信度（无噪声样本=1，噪声样本<1）(batch_size,)，默认为1
        """
        batch_size = features.shape[0]
        num_classes = centers.shape[0]
        
        # 默认为全置信（无噪声）
        if confidences is None:
            confidences = torch.ones(batch_size, device=features.device)
        
        # --------------------------
        # 类内损失：样本特征到同类中心的距离（加权MSE）
        # --------------------------
        # 获取每个样本对应的类中心 (batch_size, feature_dim)
        class_centers = centers[labels]
        # 计算距离（支持模态缺失场景：用掩码屏蔽缺失维度）
        intra_dist = torch.norm(features - class_centers, p=2, dim=1)  # (batch_size,)
        # 置信度加权（噪声样本权重低，减少对中心的干扰）
        intra_loss = torch.mean(confidences * torch.square(intra_dist))
        
        # --------------------------
        # 类间损失：不同类中心之间的距离（确保大于margin）
        # --------------------------
        # 计算所有类中心之间的距离矩阵 (num_classes, num_classes)
        center_dist = torch.cdist(centers, centers, p=2)
        # 取出上三角部分（排除对角线）
        mask = torch.triu(torch.ones(num_classes, num_classes), diagonal=1).bool()
        inter_dist = center_dist[mask]  # (num_pairs,)
        # 类间距离小于margin则产生损失
        inter_loss = torch.mean(torch.max(torch.zeros_like(inter_dist), self.margin - inter_dist))
        
        # 总损失
        total_loss = self.intra_weight * intra_loss + self.inter_weight * inter_loss
        return total_loss
