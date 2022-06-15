import torch
import torch.nn as nn

from attention import Attention
from mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        p: float = 0.,
        attn_p: float = 0.
    ):
        super().__init__()
        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.mlp = MLP(features=dim, hidden_features=int(mlp_ratio * dim), p=p)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))

        x = x + self.mlp(self.norm2(x))

        return x


if __name__ == "__main__":
    x = torch.randn((1, 16, 768))
    transformer_block = TransformerBlock(dim=768, n_heads=12)
    assert transformer_block(x).shape == x.shape
