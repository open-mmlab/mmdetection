# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .dino_layers import DinoTransformerDecoder
from .utils import MLP, inverse_sigmoid


class RTDETRTransformerDecoder(DinoTransformerDecoder):
    """Transformer decoder used in `RT-DETR.

    <https://arxiv.org/abs/2304.08069>`_.

    Args:
        eval_idx (int): The index of the decoder layer to be evaluated. If
            `eval_idx` is negative, the index will be calculated as
            `num_layers + eval_idx`. Default: -1.
    """

    def __init__(self, eval_idx: int = -1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if eval_idx < 0:
            eval_idx = self.num_layers + eval_idx
        self.eval_idx = eval_idx

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        super(DinoTransformerDecoder, self)._init_layers()
        self.ref_point_head = MLP(4, self.embed_dims * 2, self.embed_dims, 2)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                cls_branches: nn.ModuleList, **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        assert reg_branches is not None
        assert cls_branches is not None
        out_bboxes = []
        out_logits = []
        ref_points = None
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(2)
            query_pos = self.ref_point_head(reference_points)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            tmp = reg_branches[lid](query)
            assert reference_points.shape[-1] == 4
            new_reference_bboxes = tmp + inverse_sigmoid(
                reference_points, eps=1e-3)
            new_reference_bboxes = new_reference_bboxes.sigmoid()

            if self.training:
                out_logits.append(cls_branches[lid](query))
                if lid == 0:
                    out_bboxes.append(new_reference_bboxes)
                else:
                    new_bboxes = tmp + inverse_sigmoid(ref_points, eps=1e-3)
                    new_bboxes = new_bboxes.sigmoid()
                    out_bboxes.append(new_bboxes)
            elif lid == self.eval_idx:
                out_logits.append(cls_branches[lid](query))
                out_bboxes.append(new_reference_bboxes)
                break

            ref_points = new_reference_bboxes
            reference_points = new_reference_bboxes.detach(
            ) if self.training else new_reference_bboxes

        return torch.stack(out_logits), torch.stack(out_bboxes)
