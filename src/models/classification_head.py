import torch.nn as nn
import torch
import torch.nn.functional as F
from models.transformer_block import TransformerBlock

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 5,
                 hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.net(x)   # logits → we use BCEWithLogitsLoss

class TransformerFusionHead(nn.Module):
    def __init__(self, emb_dim: int = 768,
                 num_classes: int = 5,
                 num_blocks: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.0,
                 input_dropout: float = 0.1):
        super().__init__()
        # Learned positional embeddings (leads have fixed order)
        self.input_dropout = nn.Dropout(input_dropout)
        self.pos_emb = nn.Parameter(torch.zeros(1, 12, emb_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim=emb_dim,
                             num_heads=num_heads,
                             dropout=dropout,
                             attn_dropout=attn_dropout)
            for _ in range(num_blocks)
        ])
        self.per_lead_features = 64
        self.lead_dim_parser = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=256, out_features=128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=128, out_features=self.per_lead_features),
        )
        self.classifier = nn.Sequential(
            nn.Linear(12 * self.per_lead_features, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):  # x: (B, 12, 768)
        x = x + self.pos_emb
        x = self.input_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.lead_dim_parser(x)  # (B, 12, emb_dim) -> (B, 12, per_lead_features)
        x = x.flatten(1)  # (B, 12, per_lead_features) -> (B, 12 * per_lead_features)
        x = self.classifier(x)
        return x


class TransformerFusionHead_2(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=192, num_classes=5,
                 heads=8, dropout=0.1, attn_dropout=0.05):
        super().__init__()
        # 1. Project FIRST — this is the key
        self.proj = nn.Linear(emb_dim, hidden_dim)   # 768 → 192 (or 256)
        self.proj_norm = nn.LayerNorm(hidden_dim)

        # 2. Positional + dropout
        self.pos_emb = nn.Parameter(torch.randn(1, 12, hidden_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

        # 3. Single transformer block (pre-LN = stable)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=attn_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        # 4. Final classification — ONE linear layer
        self.classifier = nn.Linear(12 * hidden_dim, num_classes)

    def forward(self, x):           # (B, 12, 768)
        x = self.proj(x)            # (B, 12, 192)
        x = self.proj_norm(x)
        x = x + self.pos_emb
        x = self.dropout(x)
        x = self.transformer(x)    # (B, 12, 192)
        x = x.flatten(1)            # (B, 2304)
        return self.classifier(x)   # (B, 5)




class LeadFusionHead(nn.Module):
    def __init__(self, emb_dim=768, num_classes=5, dropout=0.3):
        super().__init__()

        self.lead_fusion_conv_1 = nn.Conv1d(12, 32, kernel_size=3, padding=1)
        self.lead_fusion_gn_1 = nn.GroupNorm(8, 32)  # ← 8 groups = 4 channels per group
        #nn.ReLU(),
        self.lead_fusion_conv_2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.lead_fusion_gn_2 = nn.GroupNorm(8, 64)
        #nn.ReLU(),
        self.lead_fusion_avg_pool_1 = nn.AdaptiveAvgPool1d(emb_dim)


        # self.lead_fusion = nn.Sequential(
        #     nn.Conv1d(12, 32, kernel_size=3, padding=1),
        #     nn.GroupNorm(8, 32),      # ← 8 groups = 4 channels per group
        #     nn.ReLU(),
        #     nn.Conv1d(32, 64, kernel_size=3, padding=1),
        #     nn.GroupNorm(8, 64),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool1d(emb_dim),
        # )
        self.classifier = nn.Sequential(
            nn.Linear(64 * emb_dim, 512),
            nn.GroupNorm(1, 512),     # ← LayerNorm-style on features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GroupNorm(1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):             # x: (B, 12, 768)
        # x = x.transpose(1, 2)         # (B, 768, 12)

        ##############
        x = self.lead_fusion_conv_1(x)      # (B, 32, 768)
        x = self.lead_fusion_gn_1(x)   # ← 8 groups = 4 channels per group
        x =  F.relu(x)
        x = self.lead_fusion_conv_2(x)     # (B, 64, 768)
        x = self.lead_fusion_gn_2(x)  # ← 8 groups = 8 channels per group
        x = F.relu(x)
        x = self.lead_fusion_avg_pool_1(x)
        ##############

        # x = self.lead_fusion(x)       # (B, 64, 768)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class ConvLeadFusionHead(nn.Module):
    def __init__(self, emb_dim=768, num_classes=5, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=3, padding=1),   # treat leads as channels
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(emb_dim),                 # back to (64, 768)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * emb_dim, 512),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):           # x: (B, 12, 768)
        x = x.transpose(1, 2)       # (B, 768, 12)
        x = self.conv(x)            # (B, 64, 768)
        x = x.flatten(1)            # (B, 64*768)
        return self.classifier(x)