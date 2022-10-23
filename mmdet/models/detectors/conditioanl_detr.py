# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict
from torch import Tensor
from .detr import DETR

import math
import warnings

import torch
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F
from mmcv.cnn import Linear, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule
from mmcv.cnn.bricks.drop import Dropout
from mmdet.utils import OptMultiConfig
from ..layers import (DetrTransformerEncoder, DetrTransformerDecoder,
                      DetrTransformerDecoderLayer, MLP, SinePositionalEncoding)
from mmdet.registry import MODELS


@MODELS.register_module()
class ConditionalDETR(DETR):
    r"""Implementation of `Conditional DETR for Fast Training Convergence.

    <https://arxiv.org/abs/2108.06152>`_.

    Code is modified from the `official github repo
    <https://github.com/Atten4Vis/ConditionalDETR>`_.
    """
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding_cfg)
        self.encoder = DetrTransformerEncoder(**self.encoder)
        self.decoder = ConditionalDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        # NOTE The embed_dims is typically passed from the inside out.
        # For example in DETR, The embed_dims is passed as
        # self_attn -> the first encoder layer -> encoder -> detector.
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

    def forward_decoder(self,
                        query: Tensor,
                        query_pos: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        memory_pos: Tensor) -> Dict:
        """Forward with Transformer decoder.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (num_query, bs, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (num_query, bs, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (num_feat, bs, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat).
            memory_pos (Tensor): The positional embeddings of memory, has
                shape (num_feat, bs, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output.#TODO
        """
        # (num_decoder_layers, num_query, bs, dim)
        hidden_states, reference_points = self.decoder(
            query=query,
            key=memory,
            value=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask)
        hidden_states = hidden_states.transpose(1, 2)
        head_inputs_dict = dict(hidden_states=hidden_states, reference_points=reference_points)
        return head_inputs_dict


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

class ConditionalDetrTransformerDecoder(DetrTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            ConditionalDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg,self.embed_dims)[1]
        #conditional detr affline
        self.query_scale = MLP(self.embed_dims, self.embed_dims, self.embed_dims, 2)
        self.ref_point_head = MLP(self.embed_dims, self.embed_dims, 2, 2)
        for layer_id in range(self.num_layers - 1):
            self.layers[layer_id + 1].cross_attn.qpos_proj = None

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                query_pos: Tensor,
                key_pos: Tensor,
                key_padding_mask: Tensor):
        reference_points_unsigmoid = self.ref_point_head(query_pos)  # [num_queries, batch_size, 2]
        reference_points = reference_points_unsigmoid.sigmoid().transpose(0, 1)
        obj_center = reference_points[..., :2].transpose(0, 1)
        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(query)
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)
            # apply transformation
            query_sine_embed = query_sine_embed * pos_transformation
            query = layer(query, key=key, value=value , query_pos=query_pos, key_pos=key_pos,
                          key_padding_mask=key_padding_mask,query_sine_embed=query_sine_embed,
                          is_first=(layer_id == 0))
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))

        if self.return_intermediate:
            return torch.stack(intermediate), reference_points

        return query, reference_points

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
                cross_attn_masks:Tensor = None,
                key_padding_mask: Tensor = None,
                query_sine_embed: Tensor = None,
                is_first=None,
                **kwargs):
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
            key_pos=key_pos,
            attn_mask=cross_attn_masks,
            key_padding_mask=key_padding_mask,
            query_sine_embed=query_sine_embed,
            is_first=is_first,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query

class ConditionalAttention(BaseModule):
    """A wrapper of conditional attention, dropout and residual connection."""

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 dropout=0.,
                 proj_drop=0.,
                 batch_first: bool = False,
                 cross_attn: bool = False,
                 keep_query_pos: bool = False,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.batch_first = batch_first  # indispensable
        self.cross_attn = cross_attn
        self.keep_query_pos = keep_query_pos
        self.embed_dims = embed_dims  # output dims
        self.num_heads = num_heads
        self.attn_dropout = Dropout(dropout)
        self.proj_drop = Dropout(proj_drop)

        self._init_proj()

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
                     need_weights: bool,
                     key_padding_mask: Tensor = None):
        assert key.size(0) == value.size(0), \
            f'{"key, value must have the same sequence length"}'
        assert query.size(1) == key.size(1) == value.size(1), \
            f'{"batch size must be equal for query, key, value"}'
        assert query.size(2) == key.size(2), \
            f'{"q_dims, k_dims must be equal"}'
        assert value.size(2) == self.embed_dims, \
            f'{"v_dims must be equal to embed_dims"}'

        tgt_len, bsz, hidden_dims = query.size()
        head_dims = hidden_dims // self.num_heads
        v_head_dims = self.embed_dims // self.num_heads
        # assert head_dims * self.num_heads == hidden_dims, \
        #     f'{"hidden_dims must be divisible by num_heads"}'
        scaling = float(head_dims) ** -0.5

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
                    bsz * self.num_heads,
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

        q = q.contiguous().view(tgt_len, bsz * self.num_heads,
                                head_dims).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads,
                                    head_dims).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads,
                                    v_head_dims).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [
            bsz * self.num_heads, tgt_len, src_len
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.attn_dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [
            bsz * self.num_heads, tgt_len, v_head_dims
        ]
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            tgt_len, bsz, self.embed_dims)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None

    def forward(
            self,
            query: Tensor,  # tgt
            key: Tensor,  # memory
            query_pos: Tensor = None,
            query_sine_embed: Tensor = None,
            key_pos: Tensor = None,  # pos
            attn_mask: Tensor = None,
            key_padding_mask: Tensor = None,  # memory_key_padding_mask
            need_weights: bool = True,
            is_first: bool = False):
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
                key_padding_mask=key_padding_mask,
                need_weights=need_weights)[0]
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
                need_weights=need_weights)[0]
            query = query + self.proj_drop(sa_output)
        return query
