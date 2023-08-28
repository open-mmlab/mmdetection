# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from ..layers import RTDETRTransformerDecoder, SinePositionalEncoding
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from .dino import DINO

DeviceType = Union[str, torch.device]


@MODELS.register_module()
class RTDETR(DINO):
    r"""Implementation of `DETRs Beat YOLOs on Real-time Object Detection
    <https://arxiv.org/abs/2304.08069>`_

    Code is modified from the `official github repo
    <https://github.com/PaddlePaddle/PaddleDetection>`_.

    backbone -> neck -> transformer -> detr_head

    Args:
        eval_size (Tuple[int, int]): The size of images for evaluation.
            Defaults to None.
        feat_strides (List[int]): The stride of each level of features.
            Defaults to [8, 16, 32].
    """

    def __init__(self,
                 *args,
                 eval_size: Tuple[int, int] = None,
                 feat_strides: List[int] = [8, 16, 32],
                 **kwargs) -> None:
        self.eval_size = eval_size
        self.eval_idx = -1
        super().__init__(*args, **kwargs)
        num_levels = self.decoder.layer_cfg.cross_attn_cfg.num_levels
        assert len(feat_strides) == len(self.backbone.out_indices)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)
        self.feat_strides = feat_strides
        assert len(self.backbone.out_indices) <= num_levels

        if self.eval_size:
            self.proposals, self.valid_mask = self.generate_proposals()

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.decoder = RTDETRTransformerDecoder(**self.decoder)
        self.embed_dims = self.decoder.embed_dims
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        """Process image features before feeding them to the transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat' and
              'spatial_shapes'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'spatial_shapes' and
                'level_start_index'.
        """
        batch_size = mlvl_feats[0].size(0)

        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for lvl, feat in enumerate(mlvl_feats):
            batch_size, c, h, w = feat.shape
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            # [num_levels, 2]
            spatial_shape = (h, w)
            # [l], start index of each level

            feat_flatten.append(feat)
            spatial_shapes.append(spatial_shape)
            level_start_index.append(h * w + level_start_index[-1])

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        level_start_index.pop()
        level_start_index = torch.as_tensor(
            level_start_index, dtype=torch.long, device=feat_flatten.device)

        # (num_level, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)

        encoder_inputs_dict = dict(
            feat=feat_flatten,
            spatial_shapes=spatial_shapes,
        )
        decoder_inputs_dict = dict(
            spatial_shapes=spatial_shapes, level_start_index=level_start_index)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat: Tensor, spatial_shapes: Tensor) -> Dict:
        """Forward with Transformer encoder. RT-DETR uses the encoder in the
        neck.

        Args:
            feat (Tensor): The input features of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).

        Returns:
            dict: The output of the Transformer encoder, which includes
            'memory', 'spatial_shapes'.
        """
        return dict(memory=feat, spatial_shapes=spatial_shapes)

    def pre_decoder(
        self,
        memory: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        if self.training or self.eval_size is None:
            output_proposals, valid_mask = self.generate_proposals(
                spatial_shapes, device=memory.device)
        else:
            output_proposals = self.proposals.to(memory.device)
            valid_mask = self.valid_mask.to(memory.device)
        original_memory = memory
        memory = torch.where(valid_mask, memory, memory.new_zeros(1))
        output_memory = self.memory_trans_fc(memory)
        output_memory = self.memory_trans_norm(output_memory)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = torch.gather(output_memory, 1,
                             topk_indices.unsqueeze(-1).repeat(1, 1, c))
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query.detach()], dim=1)
            reference_points = torch.cat(
                [dn_bbox_query, topk_coords_unact],
                dim=1).detach()  # DINO does not use detach
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=original_memory,
            reference_points=reference_points,
            dn_mask=dn_mask)

        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor = None,
                        memory_mask: Tensor = None,
                        dn_mask: Optional[Tensor] = None) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2). Defaults to None.
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Defaults to None.
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `out_logits` and `out_bboxes` of the decoder output.
        """
        out_logits, out_bboxes = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            cls_branches=self.bbox_head.cls_branches)

        decoder_outputs_dict = dict(
            hidden_states=out_logits, references=out_bboxes)
        return decoder_outputs_dict

    def generate_proposals(self,
                           spatial_shapes: Tensor = None,
                           device: Optional[torch.device] = None,
                           grid_size: float = 0.05) -> Tuple[Tensor, Tensor]:
        """Generate proposals from spatial shapes.

        Args:
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
                Defaults to None.
            device (str | torch.device): The device where the anchors will be
                put on. Defaults to None.
            grid_size (float): The grid size of the anchors. Defaults to 0.05.

        Returns:
            tuple: A tuple of proposals and valid masks.

            - proposals (Tensor): The proposals of the detector, has shape
                (bs, num_proposals, 4).
            - valid_masks (Tensor): The valid masks of the proposals, has shape
                (bs, num_proposals).
        """

        if spatial_shapes is None:
            spatial_shapes = [[
                int(self.eval_size[0] / s),
                int(self.eval_size[1] / s)
            ] for s in self.feat_strides]

        proposals = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            valid_wh = torch.tensor([H, W], dtype=torch.float32, device=device)
            grid = (grid.unsqueeze(0) + 0.5) / valid_wh
            wh = torch.ones_like(grid) * grid_size * (2.0**lvl)
            proposals.append(torch.cat((grid, wh), -1).view(-1, H * W, 4))

        proposals = torch.cat(proposals, 1)
        valid_masks = ((proposals > 0.01) * (proposals < 0.99)).all(
            -1, keepdim=True)
        proposals = torch.log(proposals / (1 - proposals))
        proposals = proposals.masked_fill(~valid_masks, float('inf'))
        return proposals, valid_masks
