# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmcv.ops import batched_nms
from mmengine.model import BaseTTAModel
from mmengine.registry import MODELS
from mmengine.structures import InstanceData

from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox_flip


def merge_aug_bboxes(aug_bboxes, aug_scores, img_metas):
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 4*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
    Returns:
        tuple[Tensor]: ``bboxes`` with shape (n,4), where
        4 represent (tl_x, tl_y, br_x, br_y)
        and ``scores`` with shape (n,).
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        ori_shape = img_info['ori_shape']
        flip = img_info['flip']
        flip_direction = img_info['flip_direction']
        if flip:
            bboxes = bbox_flip(
                bboxes=bboxes, img_shape=ori_shape, direction=flip_direction)
        recovered_bboxes.append(bboxes)
    bboxes = torch.cat(recovered_bboxes, dim=0)
    if aug_scores is None:
        return bboxes
    else:
        scores = torch.cat(aug_scores, dim=0)
        return bboxes, scores


@MODELS.register_module()
class DetTTAModel(BaseTTAModel):
    """Merge augmented detection results, only bboxes corresponding score under
    flipping and multi-scale resizing can be processed now."""

    def __init__(self, tta_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.tta_cfg = tta_cfg

    def merge_preds(self, data_samples_list: List[List[DetDataSample]]):
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (List[List[ClsDataSample]]): List of predictions
                of all enhanced data.
        Returns:
            List[ClsDataSample]: Merged prediction.
        """
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self._merge_single_sample(data_samples))
        return merged_data_samples

    def _merge_single_sample(self, data_samples):
        aug_bboxes = []
        aug_scores = []
        aug_labels = []
        img_metas = []
        for data_sample in data_samples:
            aug_bboxes.append(data_sample.pred_instances.bboxes)
            aug_scores.append(data_sample.pred_instances.scores)
            aug_labels.append(data_sample.pred_instances.labels)
            img_metas.append(data_sample.metainfo)

        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_labels = torch.cat(aug_labels, dim=0)

        if merged_bboxes.numel() == 0:
            det_bboxes = torch.cat([merged_bboxes, merged_scores[:, None]], -1)
            return [
                (det_bboxes, merged_labels),
            ]

        det_bboxes, keep_idxs = batched_nms(merged_bboxes, merged_scores,
                                            merged_labels, self.tta_cfg.nms)

        det_bboxes = det_bboxes[:self.tta_cfg.max_per_img]
        det_labels = merged_labels[keep_idxs][:self.tta_cfg.max_per_img]

        results = InstanceData()
        _det_bboxes = det_bboxes.clone()
        results.bboxes = _det_bboxes[:, :4]
        results.scores = _det_bboxes[:, 4]
        results.labels = det_labels
        det_results = data_samples[0]
        det_results.pred_instances = results
        return det_results
