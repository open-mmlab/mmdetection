# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList
from torch import Tensor

from mmdet.models.utils.vlfuse_helper import SingleScaleBiAttentionBlock
from mmdet.utils import ConfigType, OptConfigType
from .deformable_detr_layers import (DeformableDetrTransformerDecoderLayer,
                                     DeformableDetrTransformerEncoder,
                                     DeformableDetrTransformerEncoderLayer)
from .detr_layers import DetrTransformerEncoderLayer
from .dino_layers import DinoTransformerDecoder
from .utils import MLP, get_text_sine_pos_embed

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


class GroundingDinoTransformerDecoderLayer(
        DeformableDetrTransformerDecoderLayer):

    def __init__(self,
                 cross_attn_text_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 **kwargs) -> None:
        """Decoder layer of Deformable DETR."""
        self.cross_attn_text_cfg = cross_attn_text_cfg
        if 'batch_first' not in self.cross_attn_text_cfg:
            self.cross_attn_text_cfg['batch_first'] = True
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn_text = MultiheadAttention(**self.cross_attn_text_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(4)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                **kwargs) -> Tensor:
        """Implements decoder layer in Grounding DINO transformer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
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
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_attention_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        # self attention
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        # cross attention between query and text
        query = self.cross_attn_text(
            query=query,
            query_pos=query_pos,
            key=memory_text,
            value=memory_text,
            key_padding_mask=text_attention_mask)
        query = self.norms[1](query)
        # cross attention between query and image
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[2](query)
        query = self.ffn(query)
        query = self.norms[3](query)

        return query


class GroundingDinoTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output, memory_text = self.fusion_layers[layer_id](
                    visual_feature=output,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        return output, memory_text


class GroundingDinoTransformerDecoder(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)
