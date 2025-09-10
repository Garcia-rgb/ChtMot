import torch
import torch.nn as nn
import torch.nn.functional as F

from morePackage.reid_temporal import ReIDTemporalEnhancer
from morePackage.hgee_selector import HardGuidedEmbeddingEnhancer


class ReIDTemporalHGEEEnhancer(nn.Module):
    """
    集成 GICD + ConvLSTM + HGEE 的 ReID 特征增强器（多帧选择 + 平均融合）
    输入: [T, B, C, H, W] 或 [B, C, H, W]
    输出: [B, C, H, W]
    """
    def __init__(self, input_dim, hidden_dim=None, kernel_size=3, select_num=3, select_mode='max'):
        super(ReIDTemporalHGEEEnhancer, self).__init__()
        self.select_num = select_num
        self.temporal_enhancer = ReIDTemporalEnhancer(input_dim, hidden_dim, kernel_size)
        self.hgee_selector = HardGuidedEmbeddingEnhancer(select_num=select_num, mode=select_mode)

    def forward(self, x_seq):
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(0)  # [1, B, C, H, W]

        # Step 1: 时序建模增强
        enhanced_seq = self.temporal_enhancer(x_seq)  # [T, B, C, H, W]

        # Step 2: HGEE选择多帧（返回的是 [B, C, H, W]）
        if self.select_num == 1:
            return self.hgee_selector(enhanced_seq)  # 保持原逻辑
        else:
            # 修改 hgee_selector，使其返回多帧: [B, K, C, H, W]
            selected_frames = self.hgee_selector.forward_multi(enhanced_seq)  # [B, K, C, H, W]
            return selected_frames.mean(dim=1)  # 取平均，融合多帧特征
