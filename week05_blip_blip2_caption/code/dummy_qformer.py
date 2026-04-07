import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================
# 单层 Block（论文结构）
# Query 和 Text 共享 Self-Attn
# Cross-Attn 只作用在 Query 上
# ======================================
class QFormerBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, cross_attn=True):
        super().__init__()

        self.cross_attn_enabled = cross_attn

        # Self-Attn（Query + Text 一起）
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Cross-Attn（只给 Query 用）
        if cross_attn:
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, query_len, image_embeds, attn_mask=None):
        """
        hidden_states: [B, Q+T, D]
        query_len: Q
        """

        # -----------------------------
        # 1. Self-Attention（Q + T一起）
        # -----------------------------
        residual = hidden_states
        attn_out, _ = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=attn_mask
        )
        hidden_states = self.norm1(residual + attn_out)

        # -----------------------------
        # 2. Cross-Attn（只作用 Query）
        # -----------------------------
        if self.cross_attn_enabled:
            q = hidden_states[:, :query_len, :]
            residual = q

            q_attn, _ = self.cross_attn(q, image_embeds, image_embeds)
            q = self.norm2(residual + q_attn)

            hidden_states = torch.cat([q, hidden_states[:, query_len:, :]], dim=1)

        # -----------------------------
        # 3. FFN（全部 token）
        # -----------------------------
        residual = hidden_states
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.norm3(residual + hidden_states)

        return hidden_states


# ======================================
# Q-Former（完整版）
# ======================================
class QFormer(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        hidden_dim=768,
        num_queries=32,
        num_layers=12,
        num_heads=12,
        max_txt_len=64,
        cross_attn_freq=2
    ):
        super().__init__()

        self.num_queries = num_queries

        # ============================
        # Query Tokens（可学习）
        # ============================
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim)
        )

        # ============================
        # Text Embedding（论文用BERT）
        # ============================
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_txt_len, hidden_dim)

        # ============================
        # Transformer Layers
        # ============================
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            use_cross_attn = (i % cross_attn_freq == 0)
            self.layers.append(
                QFormerBlock(hidden_dim, num_heads, use_cross_attn)
            )

        self.norm = nn.LayerNorm(hidden_dim)

    # ======================================
    # 构造 Attention Mask（论文核心）
    # ======================================
    def build_attention_mask(self, query_len, text_len, mode):
        """
        mode:
            "itc" : query <-> query, text <-> text（隔离）
            "itm" : query <-> text（全连接）
            "itg" : text causal（生成）
        """
        total_len = query_len + text_len
        mask = torch.zeros(total_len, total_len)

        if mode == "itc":
            # Query 和 Text 不互相看
            mask[:query_len, query_len:] = float('-inf')
            mask[query_len:, :query_len] = float('-inf')

        elif mode == "itm":
            # 全部互相看
            pass

        elif mode == "itg":
            # 1️⃣ Text causal
            causal = torch.triu(
                torch.ones(text_len, text_len) * float('-inf'),
                diagonal=1
            )
            mask[query_len:, query_len:] = causal

            # 2️⃣ Text 不能看 Query
            mask[query_len:, :query_len] = float('-inf')

        return mask

    # ======================================
    # Forward
    # ======================================
    def forward(self, image_embeds, input_ids=None, mode="itm"):
        """
        image_embeds: [B, N, D]
        input_ids: [B, T]
        """

        B = image_embeds.size(0)

        # -----------------------------
        # Query
        # -----------------------------
        query_tokens = self.query_tokens.expand(B, -1, -1)
        query_len = query_tokens.size(1)

        # -----------------------------
        # Text
        # -----------------------------
        if input_ids is not None:
            T = input_ids.size(1)

            pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
            text_embeds = self.token_embed(input_ids) + self.pos_embed(pos_ids)

            hidden_states = torch.cat([query_tokens, text_embeds], dim=1)
            attn_mask = self.build_attention_mask(query_len, T, mode).to(input_ids.device)

        else:
            hidden_states = query_tokens
            attn_mask = None
            T = 0

        # -----------------------------
        # Transformer
        # -----------------------------
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                query_len,
                image_embeds,
                attn_mask
            )

        hidden_states = self.norm(hidden_states)

        # -----------------------------
        # 输出拆分
        # -----------------------------
        query_output = hidden_states[:, :query_len, :]

        if input_ids is not None:
            text_output = hidden_states[:, query_len:, :]
            return query_output, text_output

        return query_output


# ======================================
# Dummy Vision Encoder
# ======================================
class DummyVisionEncoder(nn.Module):
    def __init__(self, img_dim=256, hidden_dim=768):
        super().__init__()
        self.linear = nn.Linear(img_dim, hidden_dim)

    def forward(self, x):
        x = self.linear(x)
        return x.unsqueeze(1).repeat(1, 16, 1)


# ======================================
# Demo
# ======================================
def demo():
    B = 2
    img = torch.randn(B, 256)
    text = torch.randint(0, 30522, (B, 10))

    vision = DummyVisionEncoder()
    qformer = QFormer()

    image_embeds = vision(img)

    # ITC
    q, t = qformer(image_embeds, text, mode="itc")
    print("ITC:", q.shape, t.shape)

    # ITM
    q, t = qformer(image_embeds, text, mode="itm")
    print("ITM:", q.shape, t.shape)

    # ITG
    q, t = qformer(image_embeds, text, mode="itg")
    print("ITG:", q.shape, t.shape)


if __name__ == "__main__":
    demo()