import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossFrameSimilarityRefiner(nn.Module):
    """
    跨帧相似性特征重构模块
    输入特征序列 [T, B, C, H, W]
    输出优化后的特征序列 [T, B, C, H, W]
    """

    def __init__(self, feature_dim, top_k=3):
        super(CrossFrameSimilarityRefiner, self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, features):
        """
        Args:
            features: Tensor [T, B, C, H, W] - 多帧特征序列
        Returns:
            refined: Tensor [T, B, C, H, W] - 重构优化后的特征序列
        """
        if features.shape[0] == 1:
            return features[0]  # ← ✅ 直接返回 [B, C, H, W]

        T, B, C, H, W = features.shape
        features_flat = features.view(T, B, C, -1)  # → [T, B, C, HW]
        refined = []

        for t in range(T):
            ref_feat = features_flat[t]  # [B, C, HW]
            sim_scores = []

            for t2 in range(T):
                if t2 == t:
                    continue
                tgt_feat = features_flat[t2]  # [B, C, HW]
                sim = F.cosine_similarity(ref_feat.unsqueeze(2), tgt_feat.unsqueeze(1), dim=1)  # [B, HW, HW]
                sim_scores.append(sim.mean(dim=(1, 2)))  # 每个 batch 的平均相似度 [B]

            sim_tensor = torch.stack(sim_scores, dim=1)  # [B, T-1]
            topk_weights, topk_indices = torch.topk(sim_tensor, self.top_k, dim=1)

            topk_feats = []
            for b in range(B):
                selected = []
                for i in topk_indices[b]:
                    selected.append(features[i, b])  # [C, H, W]
                selected = torch.stack(selected, dim=0)  # [top_k, C, H, W]
                mean_feat = torch.mean(selected, dim=0)  # [C, H, W]
                topk_feats.append(mean_feat)

            topk_feats = torch.stack(topk_feats, dim=0)  # [B, C, H, W]
            out_feat = self.linear(topk_feats.permute(0, 2, 3, 1))  # [B, H, W, C]
            out_feat = out_feat.permute(0, 3, 1, 2)  # [B, C, H, W]
            refined.append(out_feat)

        refined = torch.stack(refined, dim=0)  # [T, B, C, H, W]
        return refined
