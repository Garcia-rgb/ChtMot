import torch
import torch.nn as nn
import torch.nn.functional as F

class HardGuidedEmbeddingEnhancer(nn.Module):
    """
    HGEE: 基于相似度的硬帧选择模块
    输入: T帧 ReID 特征序列 [T, B, C, H, W]
    输出: 被选中的一帧或多帧特征 [B, C, H, W]
    """
    def __init__(self, select_num=1, mode='max'):  # mode 可选 'max' 或 'avg'
        super(HardGuidedEmbeddingEnhancer, self).__init__()
        self.select_num = select_num
        self.mode = mode

    def forward(self, x_seq):  # x_seq: [T, B, C, H, W]
        T, B, C, H, W = x_seq.shape
        x_flat = x_seq.view(T, B, -1)  # [T, B, C*H*W]
        x_norm = F.normalize(x_flat, dim=2)  # 特征归一化

        sim = torch.einsum('tbd,sbd->bts', x_norm, x_norm)  # 计算相似度矩阵 [B, T, T]
        sim_score = sim.mean(dim=-1)  # 每帧平均相似度 [B, T]

        if self.mode == 'max':
            idx = torch.topk(sim_score, self.select_num, dim=-1)[1]  # [B, K]
        else:
            idx = torch.topk(-sim_score, self.select_num, dim=-1)[1]  # 相似度低帧（min）

        selected = []
        for b in range(B):
            frames = [x_seq[i, b] for i in idx[b]]  # [K, C, H, W]
            merged = torch.stack(frames, dim=0).mean(dim=0)  # 合并帧特征
            selected.append(merged)

        return torch.stack(selected, dim=0)  # [B, C, H, W]

    def forward_multi(self, x_seq):  # [T, B, C, H, W] → [B, K, C, H, W]
        T, B, C, H, W = x_seq.shape
        x_flat = x_seq.view(T, B, -1)
        x_norm = F.normalize(x_flat, dim=2)
        sim = torch.einsum('tbd,sbd->bts', x_norm, x_norm)
        sim_score = sim.mean(dim=-1)  # [B, T]

        # 🔔 加入日志提示
        actual_select = min(self.select_num, T)
        if self.select_num > T:
            print(f"[HGEE] Warning: select_num={self.select_num} > available_frames={T}, auto adjusted to {T}")

        # 🔍 使用实际可选帧数
        if self.mode == 'max':
            idx = torch.topk(sim_score, actual_select, dim=-1)[1]  # [B, actual_select]
        else:
            idx = torch.topk(-sim_score, actual_select, dim=-1)[1]

        selected = []
        for b in range(B):
            frames = [x_seq[i, b] for i in idx[b]]  # [actual_select, C, H, W]
            merged = torch.stack(frames, dim=0)
            selected.append(merged)

        return torch.stack(selected, dim=0)  # [B, actual_select, C, H, W]
