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
from .utils import MLP, convert_coordinate_to_encoding


class ConditionalDetrTransformerDecoder(DetrTransformerDecoder):
    """Decoder of Conditional DETR."""

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
        # we have substitute 'qpos_proj' with 'qpos_sine_proj' (exclude
        # first decoder layer),so 'qpos_proj' should be deleted.
        for layer_id in range(self.num_layers - 1):
            self.layers[layer_id + 1].cross_attn.qpos_proj = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                query_pos: Tensor, key_pos: Tensor, key_padding_mask: Tensor):
        """Forward function of decoder.

        Args:
            query (Tensor): The input query with shape
                (bs, num_queries, dim).
            key (Tensor): The input key with shape (bs, num_keys, dim) If
                `None`, the `query` will be used. Defaults to `None`.
            value (Tensor): The input value with the same shape as
                `key`. If `None`, the `key` will be used. Defaults to `None`.
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`. If not `None`, it will be added to
                `query` before forward function. Defaults to `None`.
            reg_branches (nn.Module): The regression branch for dynamically
                updating references in each layer.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            key_padding_mask (Tensor): ByteTensor with shape (bs, num_keys).
        Returns:
            List[Tensor]: forwarded results with shape (num_decoder_layers,
            bs, num_queries, dim) if `return_intermediate` is True, otherwise
            with shape (1, bs, num_queries, dim). references with shape
            (num_decoder_layers, bs, num_queries, 2).
        """
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
            ref_sine_embed = convert_coordinate_to_encoding(reference_xy)
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
    """Implements decoder layer in Conditional DETR transformer."""

    def _init_layers(self):
        """Initialize self-attention,  cross-attention, FFN, and
        normalization."""
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
            query (Tensor): The input query, has shape (bs, num_queries, dim)
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be
                added to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`.
            self_attn_masks (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), Same in `nn.MultiheadAttention.
                forward`. Defaults to None.
            cross_attn_masks (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), Same in `nn.MultiheadAttention.
                forward`. Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `cross_attn` input. ByteTensor, has shape (bs, num_keys).
            is_first (bool): A indicator to tell whether the current layer
                is the first layer of the decoder.
                Defaults to False.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
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
    """A wrapper of conditional attention, dropout and residual connection.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop: A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        cross_attn (bool): Whether the attention module is for cross attention.
            Default: False
        keep_query_pos (bool): Whether to transform query_pos before cross
            attention.
            Default: False.
        batch_first (bool): When it is True, Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default: True.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 cross_attn: bool = False,
                 keep_query_pos: bool = False,
                 batch_first: bool = True,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        assert batch_first is True, 'First \
        dimension of all DETRs in mmdet is \
        `batch`, please set `batch_first` flag.'

        self.cross_attn = cross_attn
        self.keep_query_pos = keep_query_pos
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attn_drop = Dropout(attn_drop)
        self.proj_drop = Dropout(proj_drop)

        self._init_layers()

    def _init_layers(self):
        """Initialize layers for qkv projection."""
        embed_dims = self.embed_dims
        self.qcontent_proj = Linear(embed_dims, embed_dims)
        self.qpos_proj = Linear(embed_dims, embed_dims)
        self.kcontent_proj = Linear(embed_dims, embed_dims)
        self.kpos_proj = Linear(embed_dims, embed_dims)
        self.v_proj = Linear(embed_dims, embed_dims)
        if self.cross_attn:
            self.qpos_sine_proj = Linear(embed_dims, embed_dims)
        self.out_proj = Linear(embed_dims, embed_dims)

        nn.init.constant_(self.out_proj.bias, 0.)

    def forward_attn(self,
                     query: Tensor,
                     key: Tensor,
                     value: Tensor,
                     attn_mask: Tensor,
                     key_padding_mask: Tensor = None) -> Tuple[Tensor]:
        """Forward process for `ConditionalAttention`.

        Args:
            query (Tensor): The input query with shape [bs, num_queries,
                embed_dims].
            key (Tensor): The key tensor with shape [bs, num_keys,
                embed_dims].
                If None, the `query` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tuple[Tensor]: Attention outputs of shape :math:`(N, L, E)`,
            where :math:`N` is the batch size, :math:`L` is the target
            sequence length , and :math:`E` is the embedding dimension
            `embed_dim`. Attention weights per head of shape :math:`
            (num_heads, L, S)`. where :math:`N` is batch size, :math:`L`
            is target sequence length, and :math:`S` is the source sequence
            length.
        """
        assert key.size(1) == value.size(1), \
            f'{"key, value must have the same sequence length"}'
        assert query.size(0) == key.size(0) == value.size(0), \
            f'{"batch size must be equal for query, key, value"}'
        assert query.size(2) == key.size(2), \
            f'{"q_dims, k_dims must be equal"}'
        assert value.size(2) == self.embed_dims, \
            f'{"v_dims must be equal to embed_dims"}'

        bs, tgt_len, hidden_dims = query.size()
        _, src_len, _ = key.size()
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
                if list(attn_mask.size()) != [1, query.size(1), key.size(1)]:
                    raise RuntimeError(
                        'The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                        bs * self.num_heads,
                        query.size(1),
                        key.size(1)
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

        q = q.contiguous().view(bs, tgt_len, self.num_heads,
                                head_dims).permute(0, 2, 1, 3).flatten(0, 1)
        if k is not None:
            k = k.contiguous().view(bs, src_len, self.num_heads,
                                    head_dims).permute(0, 2, 1,
                                                       3).flatten(0, 1)
        if v is not None:
            v = v.contiguous().view(bs, src_len, self.num_heads,
                                    v_head_dims).permute(0, 2, 1,
                                                         3).flatten(0, 1)

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

        attn_output_weights = F.softmax(
            attn_output_weights -
            attn_output_weights.max(dim=-1, keepdim=True)[0],
            dim=-1)
        attn_output_weights = self.attn_drop(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(
            attn_output.size()) == [bs * self.num_heads, tgt_len, v_head_dims]
        attn_output = attn_output.view(bs, self.num_heads, tgt_len,
                                       v_head_dims).permute(0, 2, 1,
                                                            3).flatten(2)
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
            query (Tensor): The input query with shape [bs, num_queries,
                embed_dims].
            key (Tensor): The key tensor with shape [bs, num_keys,
                embed_dims].
                If None, the `query` will be used. Defaults to None.
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
            [bs, num_queries, embed_dims].
        """

        if self.cross_attn:
            q_content = self.qcontent_proj(query)
            k_content = self.kcontent_proj(key)
            v = self.v_proj(key)

            bs, nq, c = q_content.size()
            _, hw, _ = k_content.size()

            k_pos = self.kpos_proj(key_pos)
            if is_first or self.keep_query_pos:
                q_pos = self.qpos_proj(query_pos)
                q = q_content + q_pos
                k = k_content + k_pos
            else:
                q = q_content
                k = k_content
            q = q.view(bs, nq, self.num_heads, c // self.num_heads)
            query_sine_embed = self.qpos_sine_proj(ref_sine_embed)
            query_sine_embed = query_sine_embed.view(bs, nq, self.num_heads,
                                                     c // self.num_heads)
            q = torch.cat([q, query_sine_embed], dim=3).view(bs, nq, 2 * c)
            k = k.view(bs, hw, self.num_heads, c // self.num_heads)
            k_pos = k_pos.view(bs, hw, self.num_heads, c // self.num_heads)
            k = torch.cat([k, k_pos], dim=3).view(bs, hw, 2 * c)
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
            q = q_content if q_pos is None else q_content + q_pos
            k = k_content if k_pos is None else k_content + k_pos
            sa_output = self.forward_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask)[0]
            query = query + self.proj_drop(sa_output)

        return query
