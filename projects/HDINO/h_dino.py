# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn.init import normal_

from mmdet.models.detectors import DINO, DeformableDETR
from mmdet.models.detectors.deformable_detr import \
    MultiScaleDeformableAttention
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType


@MODELS.register_module()
class HDINO(DINO):

    def __init__(self,
                 *args,
                 bbox_head: OptConfigType = None,
                 **kwargs) -> None:
        self.method = 0
        self.num_query_one2one = bbox_head['num_query_one2one']
        super(HDINO, self).__init__(*args, bbox_head=bbox_head, **kwargs)

    def _init_layers(self) -> None:
        super(HDINO, self)._init_layers()
        self.query_embedding = None
        if self.method == 1:
            self.query_map = nn.Linear(self.embed_dims, self.embed_dims)
        else:
            self.pos_trans_fc = nn.Linear(self.embed_dims * 2, self.embed_dims)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        super(DeformableDETR, self).init_weights()
        """Initialize weights for Transformer and other components."""
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        normal_(self.level_embed)

        if self.method == 1:
            nn.init.xavier_uniform_(self.query_map.weight)
        else:
            nn.init.xavier_uniform_(self.pos_trans_fc.weight)

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[dict, dict]:

        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
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

        # We only made changes here.
        # -------------------------------------
        if self.method == 1:
            map_memory = self.query_map(memory.detach())
            query = torch.gather(
                map_memory, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, self.embed_dims))
        else:
            pos_trans_out = self.pos_trans_fc(
                self.get_proposal_pos_embed(topk_coords_unact))
            query = self.pos_trans_norm(pos_trans_out)
        # -------------------------------------

        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()

        # We only made changes here.
        # -------------------------------------
        if self.training:
            # train: num_denoising_queries + num_query_one2one
            # + num_query_one2many
            dn_mask = decoder_inputs_dict['dn_mask']
            num_denoising_queries = head_inputs_dict['dn_meta'][
                'num_denoising_queries']
            num_query_one2one = num_denoising_queries + self.num_query_one2one
            # dn_mask[num_query_one2one:, :num_query_one2one] = True
            dn_mask[num_denoising_queries:num_query_one2one,
                    num_query_one2one:] = True
            decoder_inputs_dict['dn_mask'] = dn_mask
        else:
            # test: num_query_one2one
            # + num_query_one2many
            query = decoder_inputs_dict['query']
            reference_points = decoder_inputs_dict['reference_points']
            num_query_one2many = self.num_queries - self.num_query_one2one
            decoder_inputs_dict['query'] = query[:num_query_one2many]
            decoder_inputs_dict[
                'reference_points'] = reference_points[:num_query_one2many]
        # -------------------------------------
        return decoder_inputs_dict, head_inputs_dict
