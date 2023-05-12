# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.trackers import BaseTracker
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from .utils import mask_iou, mask_nms


@MODELS.register_module()
class IDOLTracker(BaseTracker):

    def __init__(self,
                 nms_thr_pre: float = 0.7,
                 nms_thr_post: float = 0.3,
                 init_score_thr: float = 0.2,
                 addnew_score_thr: float = 0.5,
                 obj_score_thr: float = 0.1,
                 match_score_thr: float = 0.5,
                 memo_tracklet_frames: int = 10,
                 memo_backdrop_frames: int = 1,
                 memo_momentum: float = 0.5,
                 nms_conf_thr: float = 0.5,
                 nms_backdrop_iou_thr: float = 0.5,
                 nms_class_iou_thr: float = 0.7,
                 with_cats: bool = True,
                 match_metric: str = 'bisoftmax',
                 long_match: bool = False,
                 frame_weight: bool = False,
                 temporal_weight: bool = False,
                 memory_len: int = 10,
                 **kwargs):
        super().__init__(**kwargs)
        assert 0 <= memo_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        assert memo_backdrop_frames >= 0
        self.memory_len = memory_len
        self.temporal_weight = temporal_weight
        self.long_match = long_match
        self.frame_weight = frame_weight
        self.nms_thr_pre = nms_thr_pre
        self.nms_thr_post = nms_thr_post
        self.init_score_thr = init_score_thr
        self.addnew_score_thr = addnew_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_backdrop_frames = memo_backdrop_frames
        self.memo_momentum = memo_momentum
        self.nms_conf_thr = nms_conf_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_cats = with_cats
        assert match_metric in ['bisoftmax', 'softmax', 'cosine']
        self.match_metric = match_metric

        self.reset()

    def reset(self):
        """Reset the buffer of the tracker."""
        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []

    def update(self, ids: Tensor, bboxes: Tensor, embeds: Tensor,
               labels: Tensor, frame_id: int) -> None:
        """Tracking forward function.

        Args:
            ids (Tensor): of shape(N, ).
            bboxes (Tensor): of shape (N, 5).
            embeds (Tensor): of shape (N, 256).
            labels (Tensor): of shape (N, ).
            scores (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
        """
        tracklet_inds = ids > -1

        for id, bbox, embed, label in zip(ids[tracklet_inds],
                                          bboxes[tracklet_inds],
                                          embeds[tracklet_inds],
                                          labels[tracklet_inds]):
            id = int(id)
            # update the tracked ones and initialize new tracks
            if id in self.tracks.keys():
                velocity = (bbox - self.tracks[id]['bbox']) / (
                    frame_id - self.tracks[id]['last_frame'])
                self.tracks[id]['bbox'] = bbox
                self.tracks[id]['long_score'].append(bbox[-1])
                self.tracks[id]['embed'] = (
                    1 - self.memo_momentum
                ) * self.tracks[id]['embed'] + self.memo_momentum * embed
                self.tracks[id]['long_embed'].append(embed)
                self.tracks[id]['last_frame'] = frame_id
                self.tracks[id]['label'] = label
                self.tracks[id]['velocity'] = (
                    self.tracks[id]['velocity'] * self.tracks[id]['acc_frame']
                    + velocity) / (
                        self.tracks[id]['acc_frame'] + 1)
                self.tracks[id]['acc_frame'] += 1
                self.tracks[id]['exist_frame'] += 1
            else:
                self.tracks[id] = dict(
                    bbox=bbox,
                    embed=embed,
                    label=label,
                    long_embed=[embed],
                    long_score=[bbox[-1]],
                    last_frame=frame_id,
                    velocity=torch.zeros_like(bbox),
                    acc_frame=0,
                    exist_frame=1)
        # backdrop update according to IoU
        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        self.backdrops.insert(
            0,
            dict(
                bboxes=bboxes[backdrop_inds],
                embeds=embeds[backdrop_inds],
                labels=labels[backdrop_inds]))

        # pop memo
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v['last_frame'] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
            if len(v['long_embed']) > self.memory_len:
                v['long_embed'].pop(0)
            if len(v['long_score']) > self.memory_len:
                v['long_score'].pop(0)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

    @property
    def memo(self) -> Tuple[Tensor, ...]:
        """Get tracks memory."""
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        # velocity of tracks
        memo_vs = []
        # for long term tracking
        memo_long_embeds = []
        memo_long_score = []
        memo_exist_frame = []
        # get tracks
        for k, v in self.tracks.items():
            memo_bboxes.append(v['bbox'][None, :])
            if self.long_match:
                weights = torch.stack(v['long_score'])
                if self.temporal_weight:
                    length = len(weights)
                    temporal_weight = torch.range(0.0, 1,
                                                  1 / length)[1:].to(weights)
                    weights = weights + temporal_weight
                sum_embed = (torch.stack(v['long_embed']) *
                             weights.unsqueeze(1)).sum(0) / weights.sum()
                memo_embeds.append(sum_embed[None, :])
            else:
                memo_embeds.append(v['embed'][None, :])

            memo_long_embeds.append(torch.stack(v['long_embed']))
            memo_long_score.append(torch.stack(v['long_score']))
            memo_exist_frame.append(v['exist_frame'])

            memo_ids.append(k)
            memo_labels.append(v['label'].view(1, 1))
            memo_vs.append(v['velocity'][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)
        memo_exist_frame = torch.tensor(memo_exist_frame, dtype=torch.long)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_vs = torch.cat(memo_vs, dim=0)
        return memo_bboxes, memo_labels, memo_embeds, memo_ids.squeeze(
            0), memo_vs, memo_long_embeds, memo_long_score, memo_exist_frame

    def track(self, data_sample: DetDataSample, rescale=True) -> InstanceData:

        metainfo = data_sample.metainfo
        bboxes = data_sample.pred_instances.bboxes
        masks = data_sample.pred_instances.masks
        labels = data_sample.pred_instances.labels
        scores = data_sample.pred_instances.scores
        embeds = data_sample.pred_instances.track_feats

        valids = mask_nms(masks, bboxes[:, -1], self.nms_thr_pre)
        frame_id = metainfo.get('frame_id', -1)

        bboxes = bboxes[valids, :]
        scores = scores[valids, :]
        labels = labels[valids]
        masks = masks[valids]
        embeds = embeds[valids, :]

        pred_track_instances = InstanceData()
        # # return zero bboxes if there is no track targets
        # if bboxes.shape[0] == 0:
        #     ids = torch.zeros_like(labels)
        #     pred_track_instances = data_sample.pred_det_instances.clone()
        #     pred_track_instances.instances_id = ids
        #     return pred_track_instances

        # init ids container
        ids = torch.full((bboxes.size(0), ), -2, dtype=torch.long)

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            (memo_bboxes, memo_labels, memo_embeds, memo_ids, memo_vs,
             memo_long_embeds, memo_long_score, memo_exist_frame) = self.memo

            memo_exist_frame = memo_exist_frame.to(memo_embeds)
            memo_ids = memo_ids.to(memo_embeds)

            if self.match_metric == 'bisoftmax':
                feats = torch.mm(embeds, memo_embeds.t())
                d2t_scores = feats.softmax(dim=1)
                t2d_scores = feats.softmax(dim=0)
                match_scores = (d2t_scores + t2d_scores) / 2
            elif self.match_metric == 'softmax':
                feats = torch.mm(embeds, memo_embeds.t())
                match_scores = feats.softmax(dim=1)
            elif self.match_metric == 'cosine':
                match_scores = torch.mm(
                    F.normalize(embeds, p=2, dim=1),
                    F.normalize(memo_embeds, p=2, dim=1).t())
            else:
                raise NotImplementedError
            # track according to match_scores
            for i in range(bboxes.size(0)):
                if self.frame_weight:
                    non_backs = (memo_ids > -1) & (match_scores[i, :] > 0.5)
                    if (match_scores[i, non_backs] > 0.5).sum() > 1:
                        wighted_scores = match_scores.clone()
                        frame_weight = memo_exist_frame[
                            match_scores[i, :][memo_ids > -1] > 0.5]
                        wighted_scores[i, non_backs] = wighted_scores[
                            i, non_backs] * frame_weight
                        wighted_scores[i, ~non_backs] = wighted_scores[
                            i, ~non_backs] * frame_weight.mean()
                        conf, memo_ind = torch.max(wighted_scores[i, :], dim=0)
                    else:
                        conf, memo_ind = torch.max(match_scores[i, :], dim=0)
                else:
                    conf, memo_ind = torch.max(match_scores[i, :], dim=0)
                id = memo_ids[memo_ind]
                if conf > self.match_score_thr:
                    if id > -1:
                        ids[i] = id
                        match_scores[:i, memo_ind] = 0
                        match_scores[i + 1:, memo_ind] = 0
            # initialize new tracks
            new_inds = (ids
                        == -2) & (bboxes[:, 4] > self.addnew_score_thr).cpu()
            num_news = new_inds.sum()
            ids[new_inds] = torch.arange(
                self.num_tracks, self.num_tracks + num_news, dtype=torch.long)
            self.num_tracks += num_news

            # get backdrops
            unselected_inds = torch.nonzero(
                ids == -2, as_tuple=False).squeeze(1)
            mask_ious = mask_iou(masks[unselected_inds].sigmoid() > 0.5,
                                 masks.permute(1, 0, 2, 3).sigmoid() > 0.5)
            for i, ind in enumerate(unselected_inds):
                if (mask_ious[i, :ind] < self.nms_thr_post).all():
                    ids[ind] = -1

            self.update(ids, bboxes, embeds, labels, frame_id)

        elif self.empty:
            init_inds = (ids
                         == -2) & (bboxes[:, 4] > self.init_score_thr).cpu()
            num_news = init_inds.sum()
            ids[init_inds] = torch.arange(
                self.num_tracks, self.num_tracks + num_news, dtype=torch.long)
            self.num_tracks += num_news
            unselected_inds = torch.nonzero(
                ids == -2, as_tuple=False).squeeze(1)
            mask_ious = mask_iou(masks[unselected_inds].sigmoid() > 0.5,
                                 masks.permute(1, 0, 2, 3).sigmoid() > 0.5)
            for i, ind in enumerate(unselected_inds):
                if (mask_ious[i, :ind] < self.nms_thr_post).all():
                    ids[ind] = -1
            self.update(ids, bboxes, embeds, labels, frame_id)

        tracklet_inds = ids > -1

        if rescale:
            # return result in original resolution
            # rz_*: the resize shape
            pad_height, pad_width = metainfo['pad_shape']
            rz_height, rz_width = metainfo['img_shape']
            output_h, output_w = masks.shape[-2:]
            masks = F.interpolate(
                masks,
                # size=(pad_height, pad_width),
                size=(output_h * 4, output_w * 4),
                mode='bilinear',
                align_corners=False).sigmoid()
            # crop the padding area
            masks = masks[:, :, :rz_height, :rz_width]
            ori_height, ori_width = metainfo['ori_shape']
            masks = (
                F.interpolate(
                    masks, size=(ori_height, ori_width), mode='nearest'))

        # update pred_track_instances
        pred_track_instances.bboxes = bboxes[tracklet_inds]
        pred_track_instances.masks = masks[tracklet_inds].squeeze(1) > 0.5
        # pred_track_instances.labels = labels[tracklet_inds]
        pred_track_instances.scores = scores[tracklet_inds]
        pred_track_instances.instances_id = ids[tracklet_inds]

        return pred_track_instances
