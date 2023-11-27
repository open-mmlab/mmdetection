# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.structures import InstanceData

from mmdet.models.roi_heads import StandardRoIHead
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from ...methods.builder import OVD


@MODELS.register_module()
class OVDStandardRoIHead(StandardRoIHead):

    def __init__(self, clip_cfg=None, ovd_cfg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if clip_cfg is None:
            self.clip = None
        else:
            self.clip = MODELS.build(clip_cfg)
        if ovd_cfg is not None:
            for k, v in ovd_cfg.items():
                # self.register_module(k, OVD.build(v))   # not supported in
                # pt1.8.1
                setattr(self, k, OVD.build(v))

    def _bbox_forward(self, x, rois):
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats, self.clip)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def run_ovd(self, x, batch_data_samples, rpn_results_list, ovd_name,
                batch_inputs, *args, **kwargs):
        ovd_method = getattr(self, ovd_name)

        sampling_results_list = list(
            map(ovd_method.sample, rpn_results_list, batch_data_samples))
        if isinstance(sampling_results_list[0], InstanceData):
            rois = bbox2roi([res.bboxes for res in sampling_results_list])
        else:
            sampling_results_list_ = []
            bboxes = []
            for sampling_results in sampling_results_list:
                bboxes.append(
                    torch.cat([res.bboxes for res in sampling_results]))
                sampling_results_list_ += sampling_results
            rois = bbox2roi(bboxes)
            sampling_results_list = sampling_results_list_

        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        region_embeddings = self.bbox_head.vision_to_language(bbox_feats)
        # For baron, region embeddings are pseudo words

        return ovd_method.get_losses(region_embeddings, sampling_results_list,
                                     self.clip, batch_inputs)
