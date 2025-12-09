import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim: int = 768, num_heads: int = 4, dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb=None):
        # x: (B, seq=12, emb=768)
        if pos_emb is not None:
            x = x + pos_emb
        resid = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = resid + self.dropout(x)

        resid = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = resid + self.dropout(x)
        return x