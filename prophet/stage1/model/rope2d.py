# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: A 2D version of rotary positional embeddings 
# (https://arxiv.org/abs/2104.09864).
# ------------------------------------------------------------------------------ #


import math
import torch
import torch.nn.functional as F
from torch import nn
# from einops import rearrange, repeat

def rotate_every_two(x):
    shape = x.shape
    # x = rearrange(x, '... (d j) -> ... d j', j = 2)
    # x1, x2 = x.unbind(dim = -1)
    x = x.view(*shape[:-1], -1, 2)[..., [1, 0]]
    x = x.view(*shape)
    return x

def apply_rotary_pos_emb(q, k, sinu_pos):
    sin, cos = sinu_pos
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k

# rotary embeddings for 2d position
class RoPE2d(nn.Module):
    def __init__(self, in_dim, size):
        super().__init__()
        dim = in_dim // 2
        inv_freq = 1. / (40 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, size, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        _sin = sinusoid_inp.sin()
        _cos = sinusoid_inp.cos()
        _sin, _cos = map(
            lambda x: x.unsqueeze(-1).repeat(1, 1, 2),
            (_sin, _cos)
        )
        _sin[..., 0] = -_sin[..., 0]
        _sin, _cos = map(lambda x: x.view(*x.shape[:-2], -1), (_sin, _cos))
        _sin, _cos = map(
            lambda x: torch.cat([
                x.unsqueeze(0).repeat(size, 1, 1),
                x.unsqueeze(1).repeat(1, size, 1)
            ], dim=-1).view(-1, in_dim),
            (_sin, _cos)
        )
        self.register_buffer('sin', _sin)
        self.register_buffer('cos', _cos)

    def forward(self, k, q):
        q, k = apply_rotary_pos_emb(q, k, (self.sin, self.cos))
        return q, k

if __name__ == '__main__':
    rope = RoPE2d(512, size=4)
    q = torch.randn(1, 16, 512)
    k = torch.randn(1, 16, 512)
    q, k = rope(q, k)
    print(q.shape, k.shape)
    
