# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox_overlaps
from .base_tracker import BaseTracker


@MODELS.register_module()
class MaskTrackRCNNTracker(BaseTracker):
    """Tracker for MaskTrack R-CNN.

    Args:
        match_weights (dict[str : float]): The Weighting factor when computing
        the match score. It contains keys as follows:

            - det_score (float): The coefficient of `det_score` when computing
                match score.
            - iou (float): The coefficient of `ious` when computing match
                score.
            - det_label (float): The coefficient of `label_deltas` when
                computing match score.
    """

    def __init__(self,
                 match_weights: dict = dict(
                     det_score=1.0, iou=2.0, det_label=10.0),
                 **kwargs):
        super().__init__(**kwargs)
        self.match_weights = match_weights

    def get_match_score(self, bboxes: Tensor, labels: Tensor, scores: Tensor,
                        prev_bboxes: Tensor, prev_labels: Tensor,
                        similarity_logits: Tensor) -> Tensor:
        """Get the match score.

        Args:
            bboxes (torch.Tensor): of shape (num_current_bboxes, 4) in
                [tl_x, tl_y, br_x, br_y] format. Denoting the detection
                bboxes of current frame.
            labels (torch.Tensor): of shape (num_current_bboxes, )
            scores (torch.Tensor): of shape (num_current_bboxes, )
            prev_bboxes (torch.Tensor): of shape (num_previous_bboxes, 4) in
                [tl_x, tl_y, br_x, br_y] format.  Denoting the detection bboxes
                of previous frame.
            prev_labels (torch.Tensor): of shape (num_previous_bboxes, )
            similarity_logits (torch.Tensor): of shape (num_current_bboxes,
                num_previous_bboxes + 1). Denoting the similarity logits from
                track head.

        Returns:
            torch.Tensor: The matching score of shape (num_current_bboxes,
            num_previous_bboxes + 1)
        """
        similarity_scores = similarity_logits.softmax(dim=1)

        ious = bbox_overlaps(bboxes, prev_bboxes)
        iou_dummy = ious.new_zeros(ious.shape[0], 1)
        ious = torch.cat((iou_dummy, ious), dim=1)

        label_deltas = (labels.view(-1, 1) == prev_labels).float()
        label_deltas_dummy = label_deltas.new_ones(label_deltas.shape[0], 1)
        label_deltas = torch.cat((label_deltas_dummy, label_deltas), dim=1)

        match_score = similarity_scores.log()
        match_score += self.match_weights['det_score'] * \
            scores.view(-1, 1).log()
        match_score += self.match_weights['iou'] * ious
        match_score += self.match_weights['det_label'] * label_deltas

        return match_score

    def assign_ids(self, match_scores: Tensor):
        num_prev_bboxes = match_scores.shape[1] - 1
        _, match_ids = match_scores.max(dim=1)

        ids = match_ids.new_zeros(match_ids.shape[0]) - 1
        best_match_scores = match_scores.new_zeros(num_prev_bboxes) - 1e6
        for idx, match_id in enumerate(match_ids):
            if match_id == 0:
                ids[idx] = self.num_tracks
                self.num_tracks += 1
            else:
                match_score = match_scores[idx, match_id]
                # TODO: fix the bug where multiple candidate might match
                # with the same previous object.
                if match_score > best_match_scores[match_id - 1]:
                    ids[idx] = self.ids[match_id - 1]
                    best_match_scores[match_id - 1] = match_score
        return ids, best_match_scores

    def track(self,
              model: torch.nn.Module,
              feats: List[torch.Tensor],
              data_sample: DetDataSample,
              rescale=True,
              **kwargs) -> InstanceData:
        """Tracking forward function.

        Args:
            model (nn.Module): VIS model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1 in
                MaskTrackRCNN method.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_det_instances`.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                True.

        Returns:
            :obj:`InstanceData`: Tracking results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """
        metainfo = data_sample.metainfo
        bboxes = data_sample.pred_instances.bboxes
        masks = data_sample.pred_instances.masks
        labels = data_sample.pred_instances.labels
        scores = data_sample.pred_instances.scores

        frame_id = metainfo.get('frame_id', -1)
        # create pred_track_instances
        pred_track_instances = InstanceData()

        if bboxes.shape[0] == 0:
            ids = torch.zeros_like(labels)
            pred_track_instances = data_sample.pred_instances.clone()
            pred_track_instances.instances_id = ids
            return pred_track_instances

        rescaled_bboxes = bboxes.clone()
        if rescale:
            scale_factor = rescaled_bboxes.new_tensor(
                metainfo['scale_factor']).repeat((1, 2))
            rescaled_bboxes = rescaled_bboxes * scale_factor
        roi_feats, _ = model.track_head.extract_roi_feats(
            feats, [rescaled_bboxes])

        if self.empty:
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long)
            self.num_tracks += num_new_tracks
        else:
            prev_bboxes = self.get('bboxes')
            prev_labels = self.get('labels')
            prev_roi_feats = self.get('roi_feats')

            similarity_logits = model.track_head.predict(
                roi_feats, prev_roi_feats)
            match_scores = self.get_match_score(bboxes, labels, scores,
                                                prev_bboxes, prev_labels,
                                                similarity_logits)
            ids, _ = self.assign_ids(match_scores)

        valid_inds = ids > -1
        ids = ids[valid_inds]
        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        scores = scores[valid_inds]
        masks = masks[valid_inds]
        roi_feats = roi_feats[valid_inds]

        self.update(
            ids=ids,
            bboxes=bboxes,
            labels=labels,
            scores=scores,
            masks=masks,
            roi_feats=roi_feats,
            frame_ids=frame_id)
        # update pred_track_instances
        pred_track_instances.bboxes = bboxes
        pred_track_instances.masks = masks
        pred_track_instances.labels = labels
        pred_track_instances.scores = scores
        pred_track_instances.instances_id = ids

        return pred_track_instances
