# modules/reid_temporal.py

import torch
import torch.nn as nn

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_conv = nn.Conv2d(
            input_dim + hidden_dim,  # ✅ 关键修正：拼接后的维度
            4 * hidden_dim,  # 输出4个门（i, f, o, g）
            kernel_size,
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        print(f"[ConvLSTMCell] x.shape = {x.shape}, h_prev.shape = {h_prev.shape}")
        combined = torch.cat([x, h_prev], dim=1)
        print(
            f"[ConvLSTMCell] combined.shape = {combined.shape}, expected input to conv = {self.input_conv.in_channels}")

        gates = self.input_conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ReIDTemporalEnhancer(nn.Module):
    """
    输入: T帧的ReID特征序列 [T, B, C, H, W]
    输出: T帧时序增强后的ReID特征序列 [T, B, C, H, W]
    """
    def __init__(self, input_dim, hidden_dim=None, kernel_size=3):
        super(ReIDTemporalEnhancer, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim  # 默认保持通道数不变
        self.lstm_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x_seq):  # x_seq: [T, B, C, H, W]
        T, B, C, H, W = x_seq.shape
        print(f"=== ReIDTemporalEnhancer Input Sequence Shape: {x_seq.shape}")
        h = torch.zeros(B, C, H, W, device=x_seq.device)
        c = torch.zeros(B, C, H, W, device=x_seq.device)
        output = []

        for t in range(T):
            print(f"--- Frame {t}: x_seq[{t}].shape = {x_seq[t].shape}, h.shape = {h.shape}")
            h, c = self.lstm_cell(x_seq[t], h, c)
            output.append(h)

        return torch.stack(output, dim=0)  # [T, B, C, H, W]