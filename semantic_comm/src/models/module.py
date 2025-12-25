import torch
import torch.nn as nn
import math
from transformers import BertModel

# ---- 位置编码 ----
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("channels 必须是 4 的倍数")
        self.channels = channels

    def forward(self, height, width):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        y = torch.arange(height, device=device).unsqueeze(1).repeat(1, width)
        x = torch.arange(width, device=device).unsqueeze(0).repeat(height, 1)
        div_term = torch.exp(
            torch.arange(0, self.channels // 4, device=device).float()
            * (-math.log(10000.0) / (self.channels // 4))
        )
        pos_x = torch.sin(x.unsqueeze(-1) * div_term)
        pos_y = torch.cos(y.unsqueeze(-1) * div_term)
        pos = torch.cat([pos_y, pos_x], dim=-1)
        pos = pos.permute(2, 0, 1).unsqueeze(0)
        return pos


# ---- 通用 Block，支持 proj_drop/attn_drop/drop_path 参数 ----
class Block(nn.Module):
    def __init__(
        self,
        dim,
        nheads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        depth=1
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, nheads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        self.drop_path = nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_layer(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + self.drop_path(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x
