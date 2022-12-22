# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from torch import Tensor
from torch.nn import ModuleList

from .detr_layers import DetrTransformerDecoder, DetrTransformerDecoderLayer
from .utils import MLP, ConditionalAttention, coordinate_to_encoding


class ConditionalDetrTransformerDecoder(DetrTransformerDecoder):
    """Decoder of Conditional DETR."""

    def _init_layers(self) -> None:
        """Initialize decoder layers and other layers."""
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
        # we have substitute 'qpos_proj' with 'qpos_sine_proj' except for
        # the first decoder layer), so 'qpos_proj' should be deleted
        # in other layers.
        for layer_id in range(self.num_layers - 1):
            self.layers[layer_id + 1].cross_attn.qpos_proj = None

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                key_padding_mask: Tensor = None):
        """Forward function of decoder.

        Args:
            query (Tensor): The input query with shape
                (bs, num_queries, dim).
            key (Tensor): The input key with shape (bs, num_keys, dim) If
                `None`, the `query` will be used. Defaults to `None`.
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`. If not `None`, it will be added to
                `query` before forward function. Defaults to `None`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If `None`, and `query_pos`
                has the same shape as `key`, then `query_pos` will be used
                as `key_pos`. Defaults to `None`.
            key_padding_mask (Tensor): ByteTensor with shape (bs, num_keys).
                Defaults to `None`.
        Returns:
            List[Tensor]: forwarded results with shape (num_decoder_layers,
            bs, num_queries, dim) if `return_intermediate` is True, otherwise
            with shape (1, bs, num_queries, dim). References with shape
            (bs, num_queries, 2).
        """
        reference_unsigmoid = self.ref_point_head(
            query_pos)  # [bs, num_queries, 2]
        reference = reference_unsigmoid.sigmoid()
        reference_xy = reference[..., :2]
        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(query)
            # get sine embedding for the query reference
            ref_sine_embed = coordinate_to_encoding(coord_tensor=reference_xy)
            # apply transformation
            ref_sine_embed = ref_sine_embed * pos_transformation
            query = layer(
                query,
                key=key,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                ref_sine_embed=ref_sine_embed,
                is_first=(layer_id == 0))
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))

        if self.return_intermediate:
            return torch.stack(intermediate), reference

        query = self.post_norm(query)
        return query.unsqueeze(0), reference


class ConditionalDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Implements decoder layer in Conditional DETR transformer."""

    def _init_layers(self):
        """Initialize self-attention, cross-attention, FFN, and
        normalization."""
        self.self_attn = ConditionalAttention(**self.self_attn_cfg)
        self.cross_attn = ConditionalAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_masks: Tensor = None,
                cross_attn_masks: Tensor = None,
                key_padding_mask: Tensor = None,
                ref_sine_embed: Tensor = None,
                is_first: bool = False):
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim)
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be
                added to `query` before forward function. Defaults to `None`.
            ref_sine_embed (Tensor): The positional encoding for query in
                cross attention, with the same shape as `x`. Defaults to None.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not None, it will be added to
                `key` before forward function. If None, and `query_pos` has
                the same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_masks (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), Same in `nn.MultiheadAttention.
                forward`. Defaults to None.
            cross_attn_masks (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), Same in `nn.MultiheadAttention.
                forward`. Defaults to None.
            key_padding_mask (Tensor, optional): ByteTensor, has shape
                (bs, num_keys). Defaults to None.
            is_first (bool): A indicator to tell whether the current layer
                is the first layer of the decoder. Defaults to False.

        Returns:
            Tensor: Forwarded results, has shape (bs, num_queries, dim).
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
