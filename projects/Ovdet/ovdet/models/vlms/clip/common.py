from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import MultiheadAttention
from torch.nn import functional as F


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MultiheadSelfAttention(MultiheadAttention):
    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                return_tokens: bool = False) \
            -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        assert query is value and value is key  # self-attention
        if return_tokens:
            # in_projection
            tokens = F.linear(
                value, self.in_proj_weight,
                bias=self.in_proj_bias)[..., -self.embed_dim:]
            # out_projection
            tokens = F.linear(
                tokens, self.out_proj.weight, bias=self.out_proj.bias)
        else:
            tokens = None

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask)

        return attn_output, tokens, attn_output_weights


class ResidualAttentionBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = MultiheadSelfAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, return_tokens: bool, attn_masks=None):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        else:
            self.attn_mask = None

        length = x.shape[0]

        if attn_masks is None:
            if self.attn_mask is None:
                attn_mask = None
            else:
                attn_mask = self.attn_mask[:length, :length]
        else:
            attn_mask = attn_masks

        return self.attn(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=attn_mask,
            return_tokens=return_tokens)[:2]

    def forward(self,
                x,
                return_tokens=False,
                cls_indices=None,
                attn_masks=None):
        att, tokens = self.attention(
            self.ln_1(x), return_tokens, attn_masks=attn_masks)
        if return_tokens:
            assert cls_indices is not None
            if not isinstance(cls_indices, int):
                assert len(cls_indices) == x.shape[1]  # x: LNC
            cls_tokens = x[cls_indices, torch.arange(x.shape[1])]
            tokens = cls_tokens[None] + tokens
            tokens = tokens + self.mlp(self.ln_2(tokens))

            x = x + att
            x = x + self.mlp(self.ln_2(x))

            return x, tokens
        else:
            assert tokens is None
            x = x + att
            # x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))

            return x, None


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Transformer(nn.Module):

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self,
                x: torch.Tensor,
                return_tokens=False,
                cls_indices=None,
                attn_masks=None):
        for i in range(self.layers - 1):
            x, _ = self.resblocks[i](x, attn_masks=attn_masks)
        return self.resblocks[-1](
            x,
            return_tokens=return_tokens,
            cls_indices=cls_indices,
            attn_masks=attn_masks)
