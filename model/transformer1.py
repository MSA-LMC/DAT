import torch
from einops import rearrange, repeat
from torch import nn, einsum
import math


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim),
                                 GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                                              Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_patches, input_dim, depth, heads, mlp_dim, dim_head, dropout=0.1):
        super().__init__()
        self.num_patches = num_patches
        self.input_dim = input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, input_dim))
        self.temporal_transformer = Transformer(input_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        x = x.contiguous().view(-1, self.num_patches, self.input_dim)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.temporal_transformer(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2):
        b, n, _, h = *x1.shape, self.heads
        q = self.to_q(x1)
        kv = self.to_kv(x2).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class InteractionAttentionLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.cross_attn = Residual(PreNorm(dim, CrossAttention(dim, heads, dim_head, dropout)))
        self.feed_forward = Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout)))

    def forward(self, x1, x2):
        x1 = self.cross_attn(x1, x2=x2)
        x1 = self.feed_forward(x1)
        return x1


class InteractionAttention(nn.Module):
    def __init__(self, num_patches, input_dim, depth, heads, mlp_dim, dim_head,dropout = 0.):
        super().__init__()
        self.num_patches = num_patches
        self.input_dim = input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, input_dim))
        self.layers = nn.ModuleList([
            InteractionAttentionLayer(input_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        ])

    def forward(self, x1, x2):
        x1 = x1 + self.pos_embedding
        x2 = x2 + self.pos_embedding
        for layer in self.layers:
            x1 = layer(x1, x2)
        return x1

# 实例化 TransformerEncoder
'''model = TransformerEncoder(num_patches=96, input_dim=2477, depth=1, heads=8, mlp_dim=1024, dim_head=64)
x = torch.randn(32, 96, 2477)
# 前向传播
output = model(x)
print(output.shape)

model = InteractionAttention(num_patches=96, input_dim=2477, depth=1, heads=8, mlp_dim=1024, dim_head=64)
x1 = torch.randn(32, 96, 2477)
x2 = torch.randn(32, 96, 2477)
# 前向传播
output = model(x1, x2)
print(output.shape)'''
