# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, build_norm_layer
from mmcv.cnn.bricks.drop import Dropout
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule, ModuleList
from torch import Tensor

from .detr_transformer import (DetrTransformerDecoder,
                               DetrTransformerDecoderLayer,
                               DetrTransformerEncoder,
                               DetrTransformerEncoderLayer)
from .utils import MLP, convert_coordinate_to_encoding, inverse_sigmoid


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
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
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

    def forward_attn(self, query, key, value, attn_mask,
                     key_padding_mask) -> Tuple[Tensor]:
        """Forward process for `ConditionalAttention`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims].
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tuple[Tensor]: Attention outputs of shape :math:`(L, N, E)`,
            where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E` is the
            embedding dimension ``embed_dim``. Attention weights per
            head of shape :math:`(num_heads, L, S)`. where :math:`N` is
            the batch size, :math:`L` is the target sequence length, and
            :math:`S` is the source sequence length.
        """
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

        attn_output_weights = F.softmax(
            attn_output_weights -
            attn_output_weights.max(dim=-1, keepdim=True)[0],
            dim=-1)
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

    def forward(self,
                query,
                key,
                query_pos=None,
                query_sine_embed=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                is_first=False) -> Tensor:
        """Forward function for `ConditionalAttention`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims].
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
            [num_queries, bs, embed_dims].
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
            query_sine_embed = self.qpos_sine_proj(query_sine_embed)
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


class DabDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Implements decoder layer in DAB-DETR transformer."""

    def _init_layers(self):
        """Initialize self-attention, cross-attention, FFN, normalization and
        others."""
        self.self_attn = ConditionalAttention(**self.self_attn_cfg)
        self.cross_attn = ConditionalAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
        self.keep_query_pos = self.cross_attn.keep_query_pos

    def forward(self,
                query,
                key=None,
                query_pos=None,
                query_sine_embed=None,
                key_pos=None,
                self_attn_masks=None,
                cross_attn_masks=None,
                key_padding_mask=None,
                is_first=False,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims].
                If None, the ``query`` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query in self
                attention, with the same shape as `x`. If not None,
                it will be added to `x` before forward function.
                Defaults to None.
            query_sine_embed (Tensor): The positional encoding for query in
                cross attention, with the same shape as `x`. If not None,
                it will be added to `x` before forward function.
                Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            self_attn_masks (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_masks (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
            is_first (bool): A indicator to tell whether the current layer
                is the first layer of the decoder.
                Defaults to False.

        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims].
        """

        query = self.self_attn(
            query=query,
            key=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_masks,
            **kwargs)
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            query_pos=query_pos,
            query_sine_embed=query_sine_embed,
            key_pos=key_pos,
            attn_mask=cross_attn_masks,
            key_padding_mask=key_padding_mask,
            is_first=is_first,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query


class DabDetrTransformerDecoder(DetrTransformerDecoder):
    """Decoder of DAB-DETR.

    Args:
        query_dim (int): The last dimension of query pos,
            4 for anchor format, 2 for point format.
            Defaults to 4.
        query_scale_type (str): Type of transformation applied
            to content query. Defaults to `cond_elewise`.
        modulate_hw_attn (bool): Whether to inject h&w info
            during cross conditional attention. Defaults to True.
    """

    def __init__(self,
                 *args,
                 query_dim=4,
                 query_scale_type='cond_elewise',
                 modulate_hw_attn=True,
                 **kwargs):

        self.query_dim = query_dim
        self.query_scale_type = query_scale_type
        self.modulate_hw_attn = modulate_hw_attn

        super().__init__(*args, **kwargs)

    def _init_layers(self):
        """Initialize decoder layers and other layers."""
        assert self.query_dim in [2, 4], \
            f'{"dab-detr only supports anchor prior or reference point prior"}'
        assert self.query_scale_type in [
            'cond_elewise', 'cond_scalar', 'fix_elewise'
        ]

        self.layers = ModuleList([
            DabDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        embed_dims = self.layers[0].embed_dims
        self.embed_dims = embed_dims

        self.post_norm = build_norm_layer(self.post_norm_cfg, embed_dims)[1]
        if self.query_scale_type == 'cond_elewise':
            self.query_scale = MLP(embed_dims, embed_dims, embed_dims, 2)
        elif self.query_scale_type == 'cond_scalar':
            self.query_scale = MLP(embed_dims, embed_dims, 1, 2)
        elif self.query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(self.num_layers, embed_dims)
        else:
            raise NotImplementedError('Unknown query_scale_type: {}'.format(
                self.query_scale_type))

        self.ref_point_head = MLP(self.query_dim // 2 * embed_dims, embed_dims,
                                  embed_dims, 2)

        if self.modulate_hw_attn and self.query_dim == 4:
            self.ref_anchor_head = MLP(embed_dims, embed_dims, 2, 2)

        self.keep_query_pos = self.layers[0].keep_query_pos
        if not self.keep_query_pos:
            for layer_id in range(self.num_layers - 1):
                self.layers[layer_id + 1].cross_attn.qpos_proj = None

    def forward(self,
                query,
                key,
                query_pos,
                reg_branches,
                key_pos=None,
                key_padding_mask=None,
                **kwargs) -> List[Tensor]:
        """Forward function of decoder.

        Args:
            query (Tensor): The input query with shape (num_queries, bs, dim).
            key (Tensor): The input key with shape (num_key, bs, dim) If
                `None`, the `query` will be used. Defaults to `None`.
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`. If not `None`, it will be added to
                `query` before forward function. Defaults to `None`.
            reg_branches (nn.Module): The regression branch for dynamically
                updating references in each layer.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If `None`, and `query_pos`
                has the same shape as `key`, then `query_pos` will be used
                as `key_pos`. Defaults to `None`.
            key_padding_mask (Tensor): ByteTensor with shape (bs, num_key).
                Defaults to `None`.

        Returns:
            List[Tensor]: forwarded results with shape (num_decoder_layers,
            bs, num_queries, dim) if `return_intermediate` is True, otherwise
            with shape (1, bs, num_queries, dim). references with shape
            (num_decoder_layers, bs, num_queries, 2/4) if `reg_branches`
            is not None, otherwise with shape (1, bs, num_queries, 2/4).
        """
        output = query
        reference_unsigmoid = query_pos

        reference = reference_unsigmoid.sigmoid()
        ref = [reference]

        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            obj_center = reference[..., :self.query_dim]
            query_sine_embed = convert_coordinate_to_encoding(
                pos_tensor=obj_center, num_feats=self.embed_dims // 2)
            query_pos = self.ref_point_head(
                query_sine_embed)  # [nq, bs, 2c] -> [nq, bs, c]
            # For the first decoder layer, do not apply transformation
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]
            # apply transformation
            query_sine_embed = query_sine_embed[
                ..., :self.embed_dims] * pos_transformation
            # modulated height and weight attention
            if self.modulate_hw_attn:
                assert obj_center.size(-1) == 4
                ref_hw = self.ref_anchor_head(output).sigmoid()
                query_sine_embed[..., self.embed_dims // 2:] *= \
                    (ref_hw[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., : self.embed_dims // 2] *= \
                    (ref_hw[..., 1] / obj_center[..., 3]).unsqueeze(-1)

            output = layer(
                output,
                key,
                query_pos=query_pos,
                query_sine_embed=query_sine_embed,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                is_first=(layer_id == 0),
                **kwargs)
            # iter update
            tmp = reg_branches(output)
            tmp[..., :self.query_dim] += inverse_sigmoid(reference)
            new_reference = tmp[..., :self.query_dim].sigmoid()
            if layer_id != self.num_layers - 1:
                ref.append(new_reference)
            reference = new_reference.detach()  # no grad_fn

            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(output))
                else:
                    intermediate.append(output)

        if self.post_norm is not None:
            output = self.post_norm(output)

        if self.return_intermediate:
            return [
                torch.stack(intermediate).transpose(1, 2),
                torch.stack(ref).transpose(1, 2),
            ]
        else:
            return [
                output.unsqueeze(0).transpose(
                    1, 2),  # return_intermediate is False
                torch.stack(ref).transpose(1, 2)
            ]


class DabDetrTransformerEncoder(DetrTransformerEncoder):
    """Encoder of DAB-DETR."""

    def _init_layers(self):
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        embed_dims = self.layers[0].embed_dims
        self.embed_dims = embed_dims
        self.query_scale = MLP(embed_dims, embed_dims, embed_dims, 2)

    def forward(self,
                query,
                query_pos=None,
                query_key_padding_mask=None,
                **kwargs):
        """Forward function of encoder.

        Args:
            query (Tensor): Input queries of encoder, has shape
                (num_queries, bs, dim).
            query_pos (Tensor): The positional embeddings of the queries, has
                shape (num_feat, bs, dim).
            query_key_padding_mask (Tensor): ByteTensor, the key padding mask
                of the queries, has shape (num_feat, bs).

        Returns:
            Tensor: With shape (num_queries, bs, dim).
        """

        for layer in self.layers:
            pos_scales = self.query_scale(query)
            query = layer(
                query,
                query_pos=query_pos * pos_scales,
                query_key_padding_mask=query_key_padding_mask,
                **kwargs)

        return query
