import torch
import torch.nn as nn
import torch.nn.functional as F


class TAEE(nn.Module):
    """
    Temporal-Aware Embedding Enhancement
    利用前后帧的上下文目标特征进行时序增强。
    输入：Tensor [T, B, C, H, W]
    输出：Tensor [T, B, C, H, W]
    """
    def __init__(self, feature_dim, temporal_window=3):
        super(TAEE, self).__init__()
        self.temporal_window = temporal_window
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feature_seq):
        """
        Args:
            feature_seq: [T, B, C, H, W]
        Returns:
            refined_seq: [T, B, C, H, W]
        """
        if feature_seq.shape[0] == 1:
            return feature_seq  # 无法进行时序增强

        T, B, C, H, W = feature_seq.shape
        enhanced = []

        for t in range(T):
            ref = feature_seq[t]  # 当前帧 [B, C, H, W]
            neighbors = []
            for dt in range(-self.temporal_window, self.temporal_window + 1):
                if dt == 0:
                    continue
                t_neighbor = t + dt
                if 0 <= t_neighbor < T:
                    neighbors.append(feature_seq[t_neighbor])

            if not neighbors:
                enhanced.append(ref)
                continue

            neighbors = torch.stack(neighbors, dim=0)  # [W, B, C, H, W]
            fused = torch.mean(neighbors, dim=0)  # [B, C, H, W]

            # 注意力融合当前帧和聚合帧
            attn_input = torch.cat([ref, fused], dim=1)  # [B, 2C, H, W]
            attn_weight = self.attention(attn_input)  # [B, 1, H, W]
            refined = ref * (1 - attn_weight) + fused * attn_weight  # 加权融合
            enhanced.append(refined)

        return torch.stack(enhanced, dim=0)  # [T, B, C, H, W]
