"""
https://github.com/juho-lee/set_transformer
Paper: Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks
"""
from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn
import math
from einops import rearrange


__all__ = [
    "SetAttention",
    "SelfSetAttention",
    "InducedSetAttention",
    "PoolingSetAttention",
    "IdentityKeyValuePoolingAttention",
]


class SetAttention(nn.Module):
    """
    "MAB" in the original paper
    """

    def __init__(
        self,
        dim_Q,
        dim_K,
        dim_V,
        num_heads,
        layer_norm=False,
    ):
        """
        Args:
            identity_key: do not transform K, use nn.Identity(), useful for attention
              pooling where key is the original features and we don't want to transform it.
              See CoCa paper: https://arxiv.org/abs/2205.01917
        """
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        assert self.dim_V % self.num_heads == 0
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if layer_norm:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        else:
            self.ln0 = nn.Identity()
            self.ln1 = nn.Identity()
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.act = nn.ReLU(inplace=True)

    def forward(self, Q, K, mask=None):
        """
        mask: if not none, should be (B, L_src, L_trg)
        """
        if mask is not None:
            assert mask.shape[0] == Q.shape[0]
            assert mask.shape[1] == Q.shape[1]
            assert mask.shape[2] == K.shape[1]
            # check valid mask
            assert mask.dtype == torch.bool
            assert torch.all(
                mask.sum(dim=2) > 0
            ), "each source token should attend to at least one target token"
            # repeat mask num_heads times
            mask = torch.cat([mask] * self.num_heads, 0)
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        if mask is not None:
            A.masked_fill_(mask == 0, -float("inf"))
        A = torch.softmax(A, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = self.ln0(O)
        O = O + self.act(self.fc_o(O))
        O = self.ln1(O)
        return O


class SelfSetAttention(SetAttention):
    """
    "SAB" in the original paper
    """

    def forward(self, X):
        return super().forward(X, X)


class InducedSetAttention(nn.Module):
    """
    "ISAB" in the original paper
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        num_queries,
        layer_norm=False,
    ):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_queries, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = SetAttention(
            dim_Q=dim_out,
            dim_K=dim_in,
            dim_V=dim_out,
            num_heads=num_heads,
            layer_norm=layer_norm,
        )
        self.mab1 = SetAttention(
            dim_Q=dim_in,
            dim_K=dim_out,
            dim_V=dim_out,
            num_heads=num_heads,
            layer_norm=layer_norm,
        )

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PoolingSetAttention(nn.Module):
    """
    "PMA" in the original paper
    """

    def __init__(
        self,
        dim,
        num_heads,
        num_queries,
        pool_type: Literal["avg", "concat", "none", None] = None,
        layer_norm=False,
    ):
        """
        Args:
            num_queries: pools the original set into `num_queries` features
            pool_type: 'avg', 'concat', or None
              - 'avg': average pooling, returns [B, dim]
              - 'max': max pooling, returns [B, dim]
              - 'concat': concatenate the pooled features, returns [B, num_queries*dim]
              - None: don't pool and returns [B, num_queries, dim]
        """
        super().__init__()
        assert pool_type in ["avg", "concat", "none", "max", None]
        self._pool_type = pool_type
        self.S = nn.Parameter(torch.Tensor(1, num_queries, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = SetAttention(
            dim,
            dim,
            dim,
            num_heads=num_heads,
            layer_norm=layer_norm,
        )

    def forward(self, X, mask=None):
        O = self.mab(self.S.repeat(X.size(0), 1, 1), X, mask)
        if self._pool_type == "avg":
            return O.mean(dim=1)
        elif self._pool_type == "max":
            return O.max(dim=1)[0]
        elif self._pool_type == "concat":
            return rearrange(O, "b q d -> b (q d)")
        elif self._pool_type in ["none", None]:
            return O
        else:
            raise ValueError(f"Unknown pool_type: {self._pool_type}")


class IdentityKeyValuePoolingAttention(nn.Module):
    """
    The key/value are identity functions as the original features, and only
    the query (external inducing point) is learned.
    See CoCa paper: https://arxiv.org/abs/2205.01917
    """

    def __init__(self, dim, num_heads, num_queries=1):
        """
        Args:
        """
        super().__init__()
        self.Q = nn.Parameter(torch.Tensor(1, num_queries, dim))
        nn.init.xavier_uniform_(self.Q)
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % self.num_heads == 0
        self._extra_repr = dict(dim=dim, num_heads=num_heads, num_queries=num_queries)

    def forward(self, V):
        # V: [B, L, D], L is sequence length
        B, L, D = V.size()
        assert D == self.dim
        batch_size = V.size(0)
        Q = self.Q.repeat(batch_size, 1, 1)
        K = V  # K and V are both identity functions from the original features

        dim_split = self.dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim), 2)
        O = A.bmm(V_)
        O = rearrange(O, "(nh b) q d -> b q (nh d)", b=batch_size)
        return O.mean(1)  # average over number of query vector features

    def extra_repr(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self._extra_repr.items())
