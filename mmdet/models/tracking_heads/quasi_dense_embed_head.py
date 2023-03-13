# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.task_modules import SamplingResult
from mmdet.registry import MODELS
from ..task_modules.tracking import embed_similarity
from .roi_embed_head import RoIEmbedHead


@MODELS.register_module()
class QuasiDenseEmbedHead(RoIEmbedHead):
    """The quasi-dense roi embed head.

    Args:
        embed_channels (int): The input channel of embed features.
            Defaults to 256.
        softmax_temp (int): Softmax temperature. Defaults to -1.
        loss_track (dict): The loss function for tracking. Defaults to
            MultiPosCrossEntropyLoss.
        loss_track_aux (dict): The auxiliary loss function for tracking.
            Defaults to L2Loss.
    """

    def __init__(self,
                 embed_channels: int = 256,
                 softmax_temp: int = -1,
                 loss_track: Optional[dict] = None,
                 loss_track_aux: dict = dict(
                     type='L2Loss',
                     sample_ratio=3,
                     margin=0.3,
                     loss_weight=1.0,
                     hard_mining=True),
                 init_cfg: dict = dict(
                     type='Xavier',
                     layer='Linear',
                     distribution='uniform',
                     bias=0,
                     override=dict(
                         type='Normal',
                         name='fc_embed',
                         mean=0,
                         std=0.01,
                         bias=0)),
                 **kwargs):
        super(QuasiDenseEmbedHead, self).__init__(init_cfg=init_cfg, **kwargs)

        if loss_track is None:
            loss_track = dict(
                type='MultiPosCrossEntropyLoss', loss_weight=0.25)

        self.fc_embed = nn.Linear(self.last_layer_dim, embed_channels)
        self.softmax_temp = softmax_temp
        self.loss_track = MODELS.build(loss_track)
        if loss_track_aux is not None:
            self.loss_track_aux = MODELS.build(loss_track_aux)
        else:
            self.loss_track_aux = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward the input `x`."""

        if self.num_convs > 0:
            for conv in self.convs:
                x = conv(x)
        x = x.flatten(1)
        if self.num_fcs > 0:
            for fc in self.fcs:
                x = self.relu(fc(x))
        x = self.fc_embed(x)
        return x

    def get_targets(
            self, gt_match_indices: List[Tensor],
            key_sampling_results: List[SamplingResult],
            ref_sampling_results: List[SamplingResult]) -> Tuple[List, List]:
        """Calculate the track targets and track weights for all samples in a
        batch according to the sampling_results.

        Args:
            gt_match_indices (list(Tensor)): Mapping from gt_instance_ids to
                ref_gt_instance_ids of the same tracklet in a pair of images.
            key_sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            ref_sampling_results (List[obj:SamplingResult]): Assign results of
                all reference images in a batch after sampling.

        Returns:
            Tuple[list[Tensor]]: Association results.
            Containing the following list of Tensors:

                - track_targets (list[Tensor]): The mapping instance ids from
                    all positive proposals in the key image to all proposals
                    in the reference image, each tensor in list has
                    shape (len(key_pos_bboxes), len(ref_bboxes)).
                - track_weights (list[Tensor]): Loss weights for all positive
                    proposals in a batch, each tensor in list has
                    shape (len(key_pos_bboxes),).
        """

        track_targets = []
        track_weights = []
        for _gt_match_indices, key_res, ref_res in zip(gt_match_indices,
                                                       key_sampling_results,
                                                       ref_sampling_results):
            targets = _gt_match_indices.new_zeros(
                (key_res.pos_bboxes.size(0), ref_res.bboxes.size(0)),
                dtype=torch.int)
            _match_indices = _gt_match_indices[key_res.pos_assigned_gt_inds]
            pos2pos = (_match_indices.view(
                -1, 1) == ref_res.pos_assigned_gt_inds.view(1, -1)).int()
            targets[:, :pos2pos.size(1)] = pos2pos
            weights = (targets.sum(dim=1) > 0).float()
            track_targets.append(targets)
            track_weights.append(weights)
        return track_targets, track_weights

    def match(
        self, key_embeds: Tensor, ref_embeds: Tensor,
        key_sampling_results: List[SamplingResult],
        ref_sampling_results: List[SamplingResult]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Calculate the dist matrixes for loss measurement.

        Args:
            key_embeds (Tensor): Embeds of positive bboxes in sampling results
                of key image.
            ref_embeds (Tensor): Embeds of all bboxes in sampling results
                of the reference image.
            key_sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            ref_sampling_results (List[obj:SamplingResults]): Assign results of
                all reference images in a batch after sampling.

        Returns:
            Tuple[list[Tensor]]: Calculation results.
            Containing the following list of Tensors:

                - dists (list[Tensor]): Dot-product dists between
                    key_embeds and ref_embeds, each tensor in list has
                    shape (len(key_pos_bboxes), len(ref_bboxes)).
                - cos_dists (list[Tensor]): Cosine dists between
                    key_embeds and ref_embeds, each tensor in list has
                    shape (len(key_pos_bboxes), len(ref_bboxes)).
        """

        num_key_rois = [res.pos_bboxes.size(0) for res in key_sampling_results]
        key_embeds = torch.split(key_embeds, num_key_rois)
        num_ref_rois = [res.bboxes.size(0) for res in ref_sampling_results]
        ref_embeds = torch.split(ref_embeds, num_ref_rois)

        dists, cos_dists = [], []
        for key_embed, ref_embed in zip(key_embeds, ref_embeds):
            dist = embed_similarity(
                key_embed,
                ref_embed,
                method='dot_product',
                temperature=self.softmax_temp)
            dists.append(dist)
            if self.loss_track_aux is not None:
                cos_dist = embed_similarity(
                    key_embed, ref_embed, method='cosine')
                cos_dists.append(cos_dist)
            else:
                cos_dists.append(None)
        return dists, cos_dists

    def loss(self, key_roi_feats: Tensor, ref_roi_feats: Tensor,
             key_sampling_results: List[SamplingResult],
             ref_sampling_results: List[SamplingResult],
             gt_match_indices_list: List[Tensor]) -> dict:
        """Calculate the track loss and the auxiliary track loss.

        Args:
            key_roi_feats (Tensor): Embeds of positive bboxes in sampling
                results of key image.
            ref_roi_feats (Tensor): Embeds of all bboxes in sampling results
                of the reference image.
            key_sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            ref_sampling_results (List[obj:SamplingResults]): Assign results of
                all reference images in a batch after sampling.
            gt_match_indices_list (list(Tensor)): Mapping from gt_instances_id
                to ref_gt_instances_id of the same tracklet in a pair of
                images.

        Returns:
            Dict [str: Tensor]: Calculation results.
            Containing the following list of Tensors:

                - loss_track (Tensor): Results of loss_track function.
                - loss_track_aux (Tensor): Results of loss_track_aux function.
        """
        key_track_feats = self(key_roi_feats)
        ref_track_feats = self(ref_roi_feats)

        losses = self.loss_by_feat(key_track_feats, ref_track_feats,
                                   key_sampling_results, ref_sampling_results,
                                   gt_match_indices_list)
        return losses

    def loss_by_feat(self, key_track_feats: Tensor, ref_track_feats: Tensor,
                     key_sampling_results: List[SamplingResult],
                     ref_sampling_results: List[SamplingResult],
                     gt_match_indices_list: List[Tensor]) -> dict:
        """Calculate the track loss and the auxiliary track loss.

        Args:
            key_track_feats (Tensor): Embeds of positive bboxes in sampling
                results of key image.
            ref_track_feats (Tensor): Embeds of all bboxes in sampling results
                of the reference image.
            key_sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            ref_sampling_results (List[obj:SamplingResults]): Assign results of
                all reference images in a batch after sampling.
            gt_match_indices_list (list(Tensor)): Mapping from gt_instances_id
                to ref_gt_instances_id of the same tracklet in a pair of
                images.

        Returns:
            Dict [str: Tensor]: Calculation results.
            Containing the following list of Tensors:

                - loss_track (Tensor): Results of loss_track function.
                - loss_track_aux (Tensor): Results of loss_track_aux function.
        """
        dists, cos_dists = self.match(key_track_feats, ref_track_feats,
                                      key_sampling_results,
                                      ref_sampling_results)
        targets, weights = self.get_targets(gt_match_indices_list,
                                            key_sampling_results,
                                            ref_sampling_results)
        losses = dict()

        loss_track = 0.
        loss_track_aux = 0.
        for _dists, _cos_dists, _targets, _weights in zip(
                dists, cos_dists, targets, weights):
            loss_track += self.loss_track(
                _dists, _targets, _weights, avg_factor=_weights.sum())
            if self.loss_track_aux is not None:
                loss_track_aux += self.loss_track_aux(_cos_dists, _targets)
        losses['loss_track'] = loss_track / len(dists)

        if self.loss_track_aux is not None:
            losses['loss_track_aux'] = loss_track_aux / len(dists)

        return losses

    def predict(self, bbox_feats: Tensor) -> Tensor:
        """Perform forward propagation of the tracking head and predict
        tracking results on the features of the upstream network.

        Args:
            bbox_feats: The extracted roi features.

        Returns:
            Tensor: The extracted track features.
        """
        track_feats = self(bbox_feats)
        return track_feats
