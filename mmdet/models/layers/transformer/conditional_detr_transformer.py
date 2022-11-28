# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, build_norm_layer
from mmcv.cnn.bricks.drop import Dropout
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn import ModuleList

from mmdet.utils import OptMultiConfig
from .detr_transformer import (DetrTransformerDecoder,
                               DetrTransformerDecoderLayer)
from .utils import MLP, gen_sine_embed_for_ref


class ConditionalDetrTransformerDecoder(DetrTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            ConditionalDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg,
                                          self.embed_dims)[1]
        # conditional detr affline
        self.query_scale = MLP(self.embed_dims, self.embed_dims,
                               self.embed_dims, 2)
        self.ref_point_head = MLP(self.embed_dims, self.embed_dims, 2, 2)
        for layer_id in range(self.num_layers - 1):
            self.layers[layer_id + 1].cross_attn.qpos_proj = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                query_pos: Tensor, key_pos: Tensor, key_padding_mask: Tensor):
        reference_unsigmoid = self.ref_point_head(
            query_pos)  # [num_queries, batch_size, 2]
        reference = reference_unsigmoid.sigmoid().transpose(0, 1)
        reference_xy = reference[..., :2].transpose(0, 1)
        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(query)
            # get sine embedding for the query vector
            ref_sine_embed = gen_sine_embed_for_ref(reference_xy)
            # apply transformation
            ref_sine_embed = ref_sine_embed * pos_transformation
            query = layer(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                ref_sine_embed=ref_sine_embed,
                is_first=(layer_id == 0))
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))

        if self.return_intermediate:
            return torch.stack(intermediate), reference

        return query, reference


class ConditionalDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):

    def _init_layers(self):
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = ConditionalAttention(**self.self_attn_cfg)
        self.cross_attn = ConditionalAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims  # TODO
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_masks: Tensor = None,
                cross_attn_masks: Tensor = None,
                key_padding_mask: Tensor = None,
                ref_sine_embed: Tensor = None,
                is_first=None):
        """
        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim)
                if `self.batch_first` is `False`, else (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (num_key, bs, dim)
                if `self.batch_first` is `False`, else (bs, num_key, dim).
                If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_masks (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_key), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_masks (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_key), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (num_value, bs).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (num_queries, bs, dim) if
            `self.batch_first` is `False`, else (bs, num_queries, dim).
        """
        query = self.self_attn(
            query=query,
            key=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_masks)
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_masks,
            key_padding_mask=key_padding_mask,
            ref_sine_embed=ref_sine_embed,
            is_first=is_first)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query


class ConditionalAttention(BaseModule):
    """A wrapper of conditional attention, dropout and residual connection."""

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 cross_attn: bool = False,
                 keep_query_pos: bool = False,
                 init_cfg: OptMultiConfig = None,
                 group_detr=1):
        super().__init__(init_cfg)
        self.cross_attn = cross_attn
        self.keep_query_pos = keep_query_pos
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attn_drop = Dropout(attn_drop)
        self.proj_drop = Dropout(proj_drop)

        self._init_proj()
        self.group_detr = group_detr

    def _init_proj(self):
        embed_dims = self.embed_dims
        self.qcontent_proj = Linear(embed_dims, embed_dims)
        self.qpos_proj = Linear(embed_dims, embed_dims)
        self.kcontent_proj = Linear(embed_dims, embed_dims)
        self.kpos_proj = Linear(embed_dims, embed_dims)
        self.v_proj = Linear(embed_dims, embed_dims)
        if self.cross_attn:
            self.qpos_sine_proj = Linear(embed_dims, embed_dims)
        self.out_proj = Linear(embed_dims, embed_dims)

        nn.init.constant_(self.out_proj.bias, 0.)  # init out_proj

    def forward_attn(self,
                     query: Tensor,
                     key: Tensor,
                     value: Tensor,
                     attn_mask: Tensor,
                     key_padding_mask: Tensor = None) -> Tuple[Tensor]:
        assert key.size(0) == value.size(0), \
            f'{"key, value must have the same sequence length"}'
        assert query.size(1) == key.size(1) == value.size(1), \
            f'{"batch size must be equal for query, key, value"}'
        assert query.size(2) == key.size(2), \
            f'{"q_dims, k_dims must be equal"}'
        assert value.size(2) == self.embed_dims, \
            f'{"v_dims must be equal to embed_dims"}'

        tgt_len, bs, hidden_dims = query.size()
        head_dims = hidden_dims // self.num_heads
        v_head_dims = self.embed_dims // self.num_heads
        assert head_dims * self.num_heads == hidden_dims, \
            f'{"hidden_dims must be divisible by num_heads"}'
        scaling = float(head_dims)**-0.5

        q = query * scaling
        k = key
        v = value

        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or \
                    attn_mask.dtype == torch.float64 or \
                    attn_mask.dtype == torch.float16 or \
                    attn_mask.dtype == torch.uint8 or \
                    attn_mask.dtype == torch.bool, \
                    'Only float, byte, and bool types are supported for \
                    attn_mask'

            if attn_mask.dtype == torch.uint8:
                warnings.warn('Byte tensor for attn_mask is deprecated.\
                     Use bool tensor instead.')
                attn_mask = attn_mask.to(torch.bool)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError(
                        'The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                        bs * self.num_heads,
                        query.size(0),
                        key.size(0)
                ]:
                    raise RuntimeError(
                        'The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(
                        attn_mask.dim()))
        # attn_mask's dim is 3 now.

        if key_padding_mask is not None and key_padding_mask.dtype == int:
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(tgt_len, bs * self.num_heads,
                                head_dims).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bs * self.num_heads,
                                    head_dims).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bs * self.num_heads,
                                    v_head_dims).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bs
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [
            bs * self.num_heads, tgt_len, src_len
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bs, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(
                bs * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.attn_drop(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(
            attn_output.size()) == [bs * self.num_heads, tgt_len, v_head_dims]
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            tgt_len, bs, self.embed_dims)
        attn_output = self.out_proj(attn_output)

        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bs, self.num_heads,
                                                       tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / self.num_heads

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            query_pos: Tensor = None,
            ref_sine_embed: Tensor = None,
            key_pos: Tensor = None,  # pos
            attn_mask: Tensor = None,
            key_padding_mask: Tensor = None,
            is_first: bool = False) -> Tensor:
        """Forward function for `ConditionalAttention`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query in self
                attention, with the same shape as `x`. If not None, it will
                be added to `x` before forward function.
                Defaults to None.
            query_sine_embed (Tensor): The positional encoding for query in
                cross attention, with the same shape as `x`. If not None, it
                will be added to `x` before forward function.
                Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
            is_first (bool): A indicator to tell whether the current layer
                is the first layer of the decoder.
                Defaults to False.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """
        if self.cross_attn:
            q_content = self.qcontent_proj(query)
            k_content = self.kcontent_proj(key)
            v = self.v_proj(key)

            nq, bs, c = q_content.size()
            hw, _, _ = k_content.size()

            k_pos = self.kpos_proj(key_pos)
            if is_first or self.keep_query_pos:
                q_pos = self.qpos_proj(query_pos)
                q = q_content + q_pos
                k = k_content + k_pos
            else:
                q = q_content
                k = k_content
            q = q.view(nq, bs, self.num_heads, c // self.num_heads)
            query_sine_embed = self.qpos_sine_proj(ref_sine_embed)
            query_sine_embed = query_sine_embed.view(nq, bs, self.num_heads,
                                                     c // self.num_heads)
            q = torch.cat([q, query_sine_embed], dim=3).view(nq, bs, 2 * c)
            k = k.view(hw, bs, self.num_heads, c // self.num_heads)
            k_pos = k_pos.view(hw, bs, self.num_heads, c // self.num_heads)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, 2 * c)
            ca_output = self.forward_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask)[0]
            query = query + self.proj_drop(ca_output)
        else:
            q_content = self.qcontent_proj(query)
            q_pos = self.qpos_proj(query_pos)
            k_content = self.kcontent_proj(query)
            k_pos = self.kpos_proj(query_pos)
            v = self.v_proj(query)
            num_queries, bs, _ = q_content.shape
            q = q_content if q_pos is None else q_content + q_pos
            k = k_content if k_pos is None else k_content + k_pos
            if self.training:
                q = torch.cat(
                    q.split(num_queries // self.group_detr, dim=0), dim=1)
                k = torch.cat(
                    k.split(num_queries // self.group_detr, dim=0), dim=1)
                v = torch.cat(
                    v.split(num_queries // self.group_detr, dim=0), dim=1)
            sa_output = self.forward_attn(
                query=q, key=k, value=v, attn_mask=attn_mask)[0]
            if self.training:
                sa_output = torch.cat(sa_output.split(bs, dim=1), dim=0)
            query = query + self.proj_drop(sa_output)
        return query
