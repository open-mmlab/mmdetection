# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from torch.nn import ModuleList

from mmdet.models.layers.transformer.conditional_detr_layers import (
    ConditionalDetrTransformerDecoder, ConditionalDetrTransformerDecoderLayer)
from mmdet.models.layers.transformer.utils import MLP, ConditionalAttention
from .group_conditional_attention import GroupConditionalAttention


class GroupConditionalDetrTransformerDecoder(ConditionalDetrTransformerDecoder
                                             ):
    """Decoder of Group DETR, the only change is in self.layers."""

    def _init_layers(self) -> None:
        """Initialize decoder layers and other layers."""
        self.layers = ModuleList([
            GroupConditionalDetrTransformerDecoderLayer(**self.layer_cfg)
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


class GroupConditionalDetrTransformerDecoderLayer(
        ConditionalDetrTransformerDecoderLayer):
    """Implements decoder layer in Group DETR transformer, the only change is
    in self.self_attn."""

    def _init_layers(self):
        """Initialize self-attention, cross-attention, FFN, and
        normalization."""
        self.self_attn = GroupConditionalAttention(**self.self_attn_cfg)
        self.cross_attn = ConditionalAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
