import torch
import torch.nn as nn
import math

# 这是一个基于 PyTorch 的简化 self-attention 及 cross-attention 演示代码，结构类似 Transformer
# 仅用于理解流程，不包含多头、层归一化、前馈、残差等完整 Transformer 模块。

class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # 线性层用于生成 Query、Key、Value 向量
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 最终输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim)
        B: batch size
        T: sequence length
        C: embedding dimension
        """
        B, T, C = x.shape

        # 将输入分别线性变换为 Q、K、V
        Q = self.q_proj(x)  # (B, T, C)
        K = self.k_proj(x)  # (B, T, C)
        V = self.v_proj(x)  # (B, T, C)

        # 计算注意力得分
        # K.transpose(-2, -1) 将 K 的最后两个维度交换: (B, T, C) -> (B, C, T)
        # -2 表示倒数第二个维度，-1 表示最后一个维度。
        # 注意力公式: softmax(Q * K^T / sqrt(d_k))
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(C)
        # 得到 (B, T, T) 的相似度矩阵

        # causal mask（下三角），用于自回归生成，让每个位置只关注当前位置及之前位置
        mask = torch.tril(torch.ones(T, T)).to(x.device)  # (T, T)，下三角为 1
        mask = mask.unsqueeze(0)  # (1, T, T)，同批次维度广播

        # 对不可见位置权重置为 -inf，softmax 后得到 0
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 归一化得到注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, T, T)

        # 权重加权 V 得到注意力输出
        out = torch.matmul(attn_weights, V)  # (B, T, C)

        # 输出投影
        return self.out_proj(out)
    

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # 与 self-attention 相同的 QKV 占位结构
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, context):
        """
        x: (B, T_q, C)        -> Query（例如 decoder 输入）
        context: (B, T_kv, C) -> Key & Value（例如 encoder 输出）
        """
        Q = self.q_proj(x)          # (B, T_q, C)
        K = self.k_proj(context)    # (B, T_kv, C)
        V = self.v_proj(context)    # (B, T_kv, C)

        # 计算跨注意力得分：Q 与 K^T 相乘
        # K.transpose(-2, -1) 将 K 从 (B, T_kv, C) -> (B, C, T_kv)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        # attn_scores: (B, T_q, T_kv)

        # 归一化权重
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 计算输出
        out = torch.matmul(attn_weights, V)  # (B, T_q, C)

        return self.out_proj(out)


if __name__ == "__main__":
    torch.manual_seed(42)

    B = 2
    T = 4
    C = 8

    x = torch.randn(B, T, C)          # self-attention输入
    context = torch.randn(B, T, C)    # cross-attention上下文

    self_attn = MaskedSelfAttention(C)
    cross_attn = CrossAttention(C)

    # Self-Attention
    self_out = self_attn(x)
    print("Self Attention Output shape:", self_out.shape)

    # Cross-Attention
    cross_out = cross_attn(x, context)
    print("Cross Attention Output shape:", cross_out.shape)