# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import MultiScaleDeformableAttention, batched_nms
from torch import Tensor, nn
from torch.nn.init import normal_

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.utils import OptConfigType
from ..layers import DDQTransformerDecoder
from ..utils import align_tensor
from .deformable_detr import DeformableDETR
from .dino import DINO


@MODELS.register_module()
class DDQDETR(DINO):
    r"""Implementation of `Dense Distinct Query for
    End-to-End Object Detection <https://arxiv.org/abs/2303.12776>`_

    Code is modified from the `official github repo
    <https://github.com/jshilong/DDQ>`_.

    Args:
        dense_topk_ratio (float): Ratio of num_dense queries to num_queries.
            Defaults to 1.5.
        dqs_cfg (:obj:`ConfigDict` or dict, optional): Config of
            Distinct Queries Selection. Defaults to nms with
            `iou_threshold` = 0.8.
    """

    def __init__(self,
                 *args,
                 dense_topk_ratio: float = 1.5,
                 dqs_cfg: OptConfigType = dict(type='nms', iou_threshold=0.8),
                 **kwargs):
        self.dense_topk_ratio = dense_topk_ratio
        self.decoder_cfg = kwargs['decoder']
        self.dqs_cfg = dqs_cfg
        super().__init__(*args, **kwargs)

        # a share dict in all moduls
        # pass some intermediate results and config parameters
        cache_dict = dict()
        for m in self.modules():
            m.cache_dict = cache_dict
        # first element is the start index of matching queries
        # second element is the number of matching queries
        self.cache_dict['dis_query_info'] = [0, 0]

        # mask for distinct queries in each decoder layer
        self.cache_dict['distinct_query_mask'] = []
        # pass to decoder do the dqs
        self.cache_dict['cls_branches'] = self.bbox_head.cls_branches
        # Used to construct the attention mask after dqs
        self.cache_dict['num_heads'] = self.encoder.layers[
            0].self_attn.num_heads
        # pass to decoder to do the dqs
        self.cache_dict['dqs_cfg'] = self.dqs_cfg

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        super(DDQDETR, self)._init_layers()
        self.decoder = DDQTransformerDecoder(**self.decoder_cfg)
        self.query_embedding = None
        self.query_map = nn.Linear(self.embed_dims, self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        normal_(self.level_embed)

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `memory`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
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
              `dense_topk_score`, `dense_topk_coords`,
              and `dn_meta`, when `self.training` is `True`, else is empty.
        """
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        if self.training:
            # aux dense branch particularly in DDQ DETR, which doesn't exist
            #   in DINO.
            # -1 is the aux head for the encoder
            dense_enc_outputs_class = self.bbox_head.cls_branches[-1](
                output_memory)
            dense_enc_outputs_coord_unact = self.bbox_head.reg_branches[-1](
                output_memory) + output_proposals

        topk = self.num_queries
        dense_topk = int(topk * self.dense_topk_ratio)

        proposals = enc_outputs_coord_unact.sigmoid()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        scores = enc_outputs_class.max(-1)[0].sigmoid()

        if self.training:
            # aux dense branch particularly in DDQ DETR, which doesn't exist
            #   in DINO.
            dense_proposals = dense_enc_outputs_coord_unact.sigmoid()
            dense_proposals = bbox_cxcywh_to_xyxy(dense_proposals)
            dense_scores = dense_enc_outputs_class.max(-1)[0].sigmoid()

        num_imgs = len(scores)
        topk_score = []
        topk_coords_unact = []
        # Distinct query.
        query = []

        dense_topk_score = []
        dense_topk_coords_unact = []
        dense_query = []

        for img_id in range(num_imgs):
            single_proposals = proposals[img_id]
            single_scores = scores[img_id]

            # `batched_nms` of class scores and bbox coordinations is used
            #   particularly by DDQ DETR for region proposal generation,
            #   instead of `topk` of class scores by DINO.
            _, keep_idxs = batched_nms(
                single_proposals, single_scores,
                torch.ones(len(single_scores), device=single_scores.device),
                self.cache_dict['dqs_cfg'])

            if self.training:
                # aux dense branch particularly in DDQ DETR, which doesn't
                #   exist in DINO.
                dense_single_proposals = dense_proposals[img_id]
                dense_single_scores = dense_scores[img_id]
                # sort according the score
                # Only sort by classification score, neither nms nor topk is
                #   required. So input parameter `nms_cfg` = None.
                _, dense_keep_idxs = batched_nms(
                    dense_single_proposals, dense_single_scores,
                    torch.ones(
                        len(dense_single_scores),
                        device=dense_single_scores.device), None)

                dense_topk_score.append(dense_enc_outputs_class[img_id]
                                        [dense_keep_idxs][:dense_topk])
                dense_topk_coords_unact.append(
                    dense_enc_outputs_coord_unact[img_id][dense_keep_idxs]
                    [:dense_topk])

            topk_score.append(enc_outputs_class[img_id][keep_idxs][:topk])

            # Instead of initializing the content part with transformed
            #   coordinates in Deformable DETR, we fuse the feature map
            #   embedding of distinct positions as the content part, which
            #   makes the initial queries more distinct.
            topk_coords_unact.append(
                enc_outputs_coord_unact[img_id][keep_idxs][:topk])

            map_memory = self.query_map(memory[img_id].detach())
            query.append(map_memory[keep_idxs][:topk])
            if self.training:
                # aux dense branch particularly in DDQ DETR, which doesn't
                # exist in DINO.
                dense_query.append(map_memory[dense_keep_idxs][:dense_topk])

        topk_score = align_tensor(topk_score, topk)
        topk_coords_unact = align_tensor(topk_coords_unact, topk)
        query = align_tensor(query, topk)
        if self.training:
            dense_topk_score = align_tensor(dense_topk_score)
            dense_topk_coords_unact = align_tensor(dense_topk_coords_unact)

            dense_query = align_tensor(dense_query)
            num_dense_queries = dense_query.size(1)
        if self.training:
            query = torch.cat([query, dense_query], dim=1)
            topk_coords_unact = torch.cat(
                [topk_coords_unact, dense_topk_coords_unact], dim=1)

        topk_coords = topk_coords_unact.sigmoid()
        if self.training:
            dense_topk_coords = topk_coords[:, -num_dense_queries:]
            topk_coords = topk_coords[:, :-num_dense_queries]

        topk_coords_unact = topk_coords_unact.detach()

        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)

            # Update `dn_mask` to add mask for dense queries.
            ori_size = dn_mask.size(-1)
            new_size = dn_mask.size(-1) + num_dense_queries
            new_dn_mask = dn_mask.new_ones((new_size, new_size)).bool()
            dense_mask = torch.zeros(num_dense_queries,
                                     num_dense_queries).bool()
            self.cache_dict['dis_query_info'] = [dn_label_query.size(1), topk]

            new_dn_mask[ori_size:, ori_size:] = dense_mask
            new_dn_mask[:ori_size, :ori_size] = dn_mask
            dn_meta['num_dense_queries'] = num_dense_queries
            dn_mask = new_dn_mask
            self.cache_dict['num_dense_queries'] = num_dense_queries
            self.decoder.aux_reg_branches = self.bbox_head.aux_reg_branches

        else:
            self.cache_dict['dis_query_info'] = [0, topk]
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None

        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            aux_enc_outputs_class=dense_topk_score,
            aux_enc_outputs_coord=dense_topk_coords,
            dn_meta=dn_meta) if self.training else dict()

        return decoder_inputs_dict, head_inputs_dict
