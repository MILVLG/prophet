# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: basic layers & blocks of MCAN
# ------------------------------------------------------------------------------ #

import torch
from torch import nn
from torch.nn import functional as F
import math

from .net_utils import *
from .rope2d import RoPE2d

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class MHAtt(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.__C = __C
        self.n_head = __C.MULTI_HEAD
        self.external_dim = __C.HIDDEN_SIZE
        self.internal_dim = __C.HIDDEN_SIZE // self.n_head

        self.linear_v = nn.Linear(self.external_dim, self.external_dim, bias=False)
        self.linear_k = nn.Linear(self.external_dim, self.external_dim)
        self.linear_q = nn.Linear(self.external_dim, self.external_dim)
        self.linear_merge = nn.Linear(self.external_dim, self.external_dim)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches, -1, self.n_head, self.internal_dim
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches, -1, self.n_head, self.internal_dim
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches, -1, self.n_head, self.internal_dim
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches, -1, self.external_dim
        )
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class SA_v(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.__C = __C
        self.n_head = __C.MULTI_HEAD
        self.external_dim = __C.HIDDEN_SIZE
        self.internal_dim = __C.HIDDEN_SIZE // self.n_head

        self.linear_v = nn.Linear(self.external_dim, self.external_dim, bias=False)
        self.linear_k = nn.Linear(self.external_dim, self.external_dim)
        self.linear_q = nn.Linear(self.external_dim, self.external_dim)
        self.linear_merge = nn.Linear(self.external_dim, self.external_dim)

        self.dropout = nn.Dropout(__C.DROPOUT_R)


        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = nn.LayerNorm(__C.HIDDEN_SIZE)
        self.rope = RoPE2d(self.internal_dim, __C.IMG_FEAT_GRID)

    def forward(self, *args):
        x, *_ = args
        n_batches = x.size(0)

        v = self.linear_v(x).view(
            n_batches, -1, self.n_head, self.internal_dim
        ).transpose(1, 2)

        k = self.linear_k(x).view(
            n_batches, -1, self.n_head, self.internal_dim
        ).transpose(1, 2)

        q = self.linear_q(x).view(
            n_batches, -1, self.n_head, self.internal_dim
        ).transpose(1, 2)

        q, k = self.rope(q, k)

        atted = self.att(v, k, q, None)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches, -1, self.external_dim
        )
        atted = self.linear_merge(atted)

        x = self.norm1(x + self.dropout1(atted))

        return x

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = nn.LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, *args):
        x = self.norm1(x + self.dropout1(
            self.mlp(x)
        ))
        return x


class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = nn.LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask, *args):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        return x


class GA(nn.Module):
    def __init__(self, __C):
        super().__init__()

        self.mhatt1 = MHAtt(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = nn.LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask, *args):

        x = self.norm1(x + self.dropout1(
            self.mhatt1(y, y, x, y_mask)
        ))

        return x