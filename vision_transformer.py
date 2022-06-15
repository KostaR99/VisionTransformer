import torch
import torch.nn as nn

from embeding import PatchEmbeding
from transformer_block import TransformerBlock


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        n_classes: int,
        depth: int,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        p: float,
        attn_p: float
    ):
        super().__init__()
        self.patch_embeding = PatchEmbeding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        self.cls_token = nn.parameter.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_enc = nn.parameter.Parameter(torch.zeros(1, 1 + self.patch_embeding.n_patches, embed_dim))

        self.pos_drop = nn.Dropout(p)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embeding(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_enc
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)

        return x


if __name__ == "__main__":
    x = torch.randn((1, 3, 64, 64))
    model = VisionTransformer(
        img_size=64,
        patch_size=16,
        in_chans=3,
        n_classes=1,
        depth=1,
        embed_dim=768,
        n_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        p=0.,
        attn_p=0.
    )

    assert model(x).shape == (1, 1)
