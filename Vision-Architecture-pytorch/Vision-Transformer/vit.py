import torch
import torch.nn as nn
import numpy as np

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        dim : int, 
        num_heads : int,
        qkv_bias : bool = False,
        attn_drop : float = 0.,
        proj_drop : float = 0.
    ) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # assert C % self.num_heads == 0, "D of x should be divisible by num_heads"
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn_probs = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_probs)

        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_probs



class Mlp(nn.Module):
    def __init__(
        self,
        in_features : int,
        hidden_features : int,
        out_features : int,
        act_layer = nn.GELU,
        bias : bool = True,
        drop : float = 0.,
    ) -> None:
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x



class Block(nn.Module):
    def __init__(
        self,
        dim : int,
        num_heads : int,
        mlp_ratio : int = 4,
        qkv_bias : bool = False,
        attn_drop : float = 0.,
        proj_drop : float = 0.,
        act_layer = nn.GELU,
        norm_layer = nn.LayerNorm,
    ) -> None:
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads, qkv_bias, attn_drop, proj_drop
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim*mlp_ratio), dim, act_layer)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()