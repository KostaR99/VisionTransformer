import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 12, qkv_bias: bool = True, attn_p: float = 0., proj_p: float = 0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, embed_dim = x.shape
        qkv = self.qkv(x)  # n_samples, n_tokens, embed_dim * 3
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, n_samples, n_heads, n_tokens, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_t = k.transpose(-2, -1)
        dp = (q @ k_t) * self.scale
        attention = dp.softmax(dim=-1)
        attention = self.attn_drop(attention)
        weighted_avg = attention @ v  # n_samples, n_heads, n_patches, head_dim
        weighted_avg = weighted_avg.transpose(1, 2)
        # n_samples, n_patches, n_heads, head_dim
        weighted_avg = weighted_avg.flatten(2)  # n_samples, n_heads, dim
        projection = self.projection(weighted_avg)
        projection = self.projection_dropout(projection)

        return projection

class AttentionSeparateQKV(nn.Module):
    def __init__(self, dim: int, n_heads: int = 12, qkv_bias: bool = True, attn_p: float = 0., proj_p: float = 0.) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.Qm = nn.Linear(dim, dim, bias=qkv_bias)
        self.Km = nn.Linear(dim, dim, bias=qkv_bias)
        self.Vm = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.projection = nn.Linear(dim, dim)
        self.projection_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, _ = x.shape
        Q = self.Qm(x)  # n_samples, n_tokens, embed_dim
        K = self.Km(x)  # n_samples, n_tokens, embed_dim
        V = self.Km(x)  # n_samples, n_tokens, embed_dim

        Q = Q.reshape(n_samples, n_tokens, self.n_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3)  # n_samples, n_heads, n_tokens, head_dim

        K = K.reshape(n_samples, n_tokens, self.n_heads, self.head_dim)
        K = K.permute(0, 2, 1, 3)  # n_samples, n_heads, n_tokens, head_dim

        V = V.reshape(n_samples, n_tokens, self.n_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3)  # n_samples, n_heads, n_tokens, head_dim

        K_t = K.transpose(-2, -1)
        dp = (Q @ K_t) * self.scale
        attention = dp.softmax(dim=-1)
        attention = self.attn_drop(attention)
        weighted_avg = attention @ V # n_samples, n_heads, n_patches, head_dim
        weighted_avg = weighted_avg.transpose(1, 2) # n_samples, n_patches, n_heads, head_dim
        weighted_avg = weighted_avg.flatten(2) # n_samples, n_patches, embed_dim
        x = self.projection(weighted_avg)
        x = self.projection_drop(x)
        return x


if __name__ == "__main__":
    x = torch.randn((1, 16, 768))
    attention = AttentionSeparateQKV(dim=768)
    assert attention(x).shape == x.shape
