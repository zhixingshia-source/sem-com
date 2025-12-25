import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange
from transformers.models.bert import BertConfig
from .module import Block, BertModel


class SetokDeTokenizer(nn.Module):
    def __init__(
        self,
        token_feat_dim: int = 768,
        hidden_dim: int = 768,
        patch_size: int = 14,
        image_size: int = 224,
        decoder_embed_dim: int = 512,
        decoder_nheads: int = 2,
        proj_drop: float = 0.1,
        attn_drop: float = 0.1,
        decoder_depth: int = 2,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_ratio: float = 4.0,
        feature_mapper_path_or_name: str = "bert-base-uncased",
        num_hidden_layers: int = 1,
        cross_attention_freq: int = 1,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__()

        self.token_feat_dim = token_feat_dim
        self.hidden_dim = hidden_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_mask_token = (image_size // patch_size) ** 2

        # =====================
        # 基础层
        # =====================
        self.mask_tokens = nn.Parameter(torch.zeros(1, self.num_mask_token, hidden_dim))
        self.mask_tokens.data.normal_(mean=0.0, std=initializer_range)

        self.mapper_fc_in = nn.Linear(token_feat_dim, hidden_dim)
        self.decoder_fc_in = nn.Linear(hidden_dim, decoder_embed_dim)
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pixel_decoder = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    nheads=decoder_nheads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    depth=1,
                )
                for _ in range(decoder_depth)
            ]
        )

        # ✅ 修复点：显式 2D 位置编码映射
        self.position_embedding = nn.Linear(2, decoder_embed_dim)

        # ✅ 修复点：最终输出到 RGB（3 通道）
        self.to_rgb = nn.Sequential(
            nn.Conv2d(decoder_embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=1),
            nn.Sigmoid(),  # 输出 0-1
        )

        # feature mapper
        self.init_feature_mapper(
            feature_mapper_path_or_name,
            hidden_dim,
            self.num_mask_token,
            num_hidden_layers,
            cross_attention_freq,
        )

        self._init_weights()

    # =====================
    # 初始化权重
    # =====================
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def init_feature_mapper(
        self,
        feature_mapper_path_or_name: str,
        vision_width: int,
        num_mask_token: int,
        num_hidden_layers: int,
        cross_attention_freq: int,
    ):
        print(f"feature_mapper_path_or_name: {feature_mapper_path_or_name}")
        mapper_config = BertConfig.from_pretrained(feature_mapper_path_or_name)
        mapper_config.encoder_width = vision_width
        mapper_config.add_cross_attention = True
        mapper_config.cross_attention_freq = cross_attention_freq
        mapper_config.query_length = num_mask_token
        mapper_config.num_hidden_layers = num_hidden_layers
        mapper_config.is_decoder = True
        mapper_config.use_cache = False
        self.mapper = BertModel.from_pretrained(feature_mapper_path_or_name, config=mapper_config)

    # =====================
    # 前向传播
    # =====================
    def forward(
        self,
        x: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        width: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        B, N, D = x.shape

        # 自动推断 height/width
        if width is None:
            width = int(N ** 0.5)
        height = max(1, N // width)
        if height * width < N:
            height += 1

        # 特征映射
        mask_tokens = self.mask_tokens.expand(B, -1, -1)
        x = self.mapper_fc_in(x)
        x = self.mapper(
            inputs_embeds=mask_tokens[:, :N, :],
            encoder_hidden_states=x,
            encoder_attention_mask=attention_masks,
            return_dict=True,
        ).last_hidden_state

        x = self.decoder_fc_in(x)
        x = self.decoder_norm(x)

        # 2D 位置编码
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=x.device),
            torch.linspace(-1, 1, width, device=x.device),
            indexing="ij",
        )
        pos = torch.stack([grid_x, grid_y], dim=-1).reshape(1, height * width, 2)
        pos_emb = self.position_embedding(pos)
        pos_emb = pos_emb[:, : x.shape[1], :]
        x = x + pos_emb

        # transformer 解码
        for block in self.pixel_decoder:
            x = block(x)
        x = self.decoder_norm(x)

        # reshape 成特征图
        total_tokens = height * width
        if total_tokens > x.shape[1]:
            pad = total_tokens - x.shape[1]
            x = torch.cat(
                [x, torch.zeros(B, pad, x.shape[2], device=x.device)], dim=1
            )
        x = rearrange(x[:, : total_tokens, :], "B (h w) C -> B C h w", h=height, w=width)

        # ✅ 转为 RGB 输出
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.to_rgb(x)

        return x
