# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from mmcv.ops import batched_nms
from torch import Tensor, nn

from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from .deformable_detr_layers import DeformableDetrTransformerDecoder
from .utils import MLP, coordinate_to_encoding, inverse_sigmoid


class DDQTransformerDecoder(DeformableDetrTransformerDecoder):
    """Transformer decoder of DDQ."""

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        super()._init_layers()
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def select_distinct_queries(self, reference_points: Tensor, query: Tensor,
                                self_attn_mask: Tensor, layer_index):
        """Get updated `self_attn_mask` for distinct queries selection, it is
        used in self attention layers of decoder.

        Args:
            reference_points (Tensor): The input reference of decoder,
                has shape (bs, num_queries, 4) with the last dimension
                arranged as (cx, cy, w, h).
            query (Tensor): The input query of decoder, has shape
                (bs, num_queries, dims).
            self_attn_mask (Tensor): The input self attention mask of
                last decoder layer, has shape (bs, num_queries_total,
                num_queries_total).
            layer_index (int): Last decoder layer index, used to get
                classification score of last layer output, for
                distinct queries selection.

        Returns:
            Tensor: `self_attn_mask` used in self attention layers
                of decoder, has shape (bs, num_queries_total,
                num_queries_total).
        """
        num_imgs = len(reference_points)
        dis_start, num_dis = self.cache_dict['dis_query_info']
        # shape of self_attn_mask
        # (batch⋅num_heads, num_queries, embed_dims)
        dis_mask = self_attn_mask[:, dis_start:dis_start + num_dis,
                                  dis_start:dis_start + num_dis]
        # cls_branches from DDQDETRHead
        scores = self.cache_dict['cls_branches'][layer_index](
            query[:, dis_start:dis_start + num_dis]).sigmoid().max(-1).values
        proposals = reference_points[:, dis_start:dis_start + num_dis]
        proposals = bbox_cxcywh_to_xyxy(proposals)

        attn_mask_list = []
        for img_id in range(num_imgs):
            single_proposals = proposals[img_id]
            single_scores = scores[img_id]
            attn_mask = ~dis_mask[img_id * self.cache_dict['num_heads']][0]
            # distinct query inds in this layer
            ori_index = attn_mask.nonzero().view(-1)
            _, keep_idxs = batched_nms(single_proposals[ori_index],
                                       single_scores[ori_index],
                                       torch.ones(len(ori_index)),
                                       self.cache_dict['dqs_cfg'])

            real_keep_index = ori_index[keep_idxs]

            attn_mask = torch.ones_like(dis_mask[0]).bool()
            # such a attn_mask give best result
            # If it requires to keep index i, then all cells in row or column
            #   i should be kept in `attn_mask` . For example, if
            #   `real_keep_index` = [1, 4], and `attn_mask` size = [8, 8],
            #   then all cells at rows or columns [1, 4] should be kept, and
            #   all the other cells should be masked out. So the value of
            #  `attn_mask` should be:
            #
            # target\source   0 1 2 3 4 5 6 7
            #             0 [ 0 1 0 0 1 0 0 0 ]
            #             1 [ 1 1 1 1 1 1 1 1 ]
            #             2 [ 0 1 0 0 1 0 0 0 ]
            #             3 [ 0 1 0 0 1 0 0 0 ]
            #             4 [ 1 1 1 1 1 1 1 1 ]
            #             5 [ 0 1 0 0 1 0 0 0 ]
            #             6 [ 0 1 0 0 1 0 0 0 ]
            #             7 [ 0 1 0 0 1 0 0 0 ]
            attn_mask[real_keep_index] = False
            attn_mask[:, real_keep_index] = False

            attn_mask = attn_mask[None].repeat(self.cache_dict['num_heads'], 1,
                                               1)
            attn_mask_list.append(attn_mask)
        attn_mask = torch.cat(attn_mask_list)
        self_attn_mask = copy.deepcopy(self_attn_mask)
        self_attn_mask[:, dis_start:dis_start + num_dis,
                       dis_start:dis_start + num_dis] = attn_mask
        # will be used in loss and inference
        self.cache_dict['distinct_query_mask'].append(~attn_mask)
        return self_attn_mask

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tensor:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries,
                dims).
            value (Tensor): The input values, has shape (bs, num_value, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups, distinct queries and
                dense queries, has shape (num_queries_total,
                num_queries_total). It will be updated for distinct queries
                selection in this forward function. It is `None` when
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
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (bs, num_queries, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, bs, num_queries,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4) when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (1 + num_decoder_layers, bs, num_queries, 4).
              The coordinates are arranged as (cx, cy, w, h).
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        self.cache_dict['distinct_query_mask'] = []
        if self_attn_mask is None:
            self_attn_mask = torch.zeros((query.size(1), query.size(1)),
                                         device=query.device).bool()
        # shape is (batch*number_heads, num_queries, num_queries)
        self_attn_mask = self_attn_mask[None].repeat(
            len(query) * self.cache_dict['num_heads'], 1, 1)
        for layer_index, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :],
                num_feats=self.embed_dims // 2)
            query_pos = self.ref_point_head(query_sine_embed)

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

            if not self.training:
                tmp = reg_branches[layer_index](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                if layer_index < (len(self.layers) - 1):
                    self_attn_mask = self.select_distinct_queries(
                        reference_points, query, self_attn_mask, layer_index)

            else:
                num_dense = self.cache_dict['num_dense_queries']
                tmp = reg_branches[layer_index](query[:, :-num_dense])
                tmp_dense = self.aux_reg_branches[layer_index](
                    query[:, -num_dense:])

                tmp = torch.cat([tmp, tmp_dense], dim=1)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                if layer_index < (len(self.layers) - 1):
                    self_attn_mask = self.select_distinct_queries(
                        reference_points, query, self_attn_mask, layer_index)

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points
