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
        img_shapes (list[Tensor]): shape (3, ).
    Returns:
        tuple[Tensor]: ``bboxes`` with shape (n,4), where
        4 represent (tl_x, tl_y, br_x, br_y)
        and ``scores`` with shape (n,).
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        ori_shape = img_info[0]['ori_shape']
        flip = img_info[0]['flip']
        flip_direction = img_info[0]['flip_direction']
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


def merge_aug_masks(aug_masks, img_metas):
    """Merge augmented mask prediction.

    Args:
        aug_masks (list[Tensor]): each has shape
            (n, c, h, w).
        img_metas (dict): Image information.
        weights (list or Tensor): Weight of each aug_masks,
            the length should be n.

    Returns:
        Tensor: has shape (n, c, h, w)
    """
    recovered_masks = []
    for mask, img_info in zip(aug_masks, img_metas):
        flip = img_info[0]['flip']
        flip_direction = img_info[0]['flip_direction']
        if flip:
            if flip_direction == 'horizontal':
                mask = mask.flip(-1)
            elif flip_direction == 'vertical':
                mask = mask.flip(1)
            elif flip_direction == 'diagonal':
                mask = mask.flip(-1).flip(1)
            else:
                raise ValueError(
                    f"Invalid flipping direction '{flip_direction}'")
        recovered_masks.append(mask[None, :].float())

    merged_masks = torch.cat(recovered_masks, 0).mean(dim=0)
    return merged_masks


@MODELS.register_module()
class DetTTAModel(BaseTTAModel):

    def __init__(self, tta_cfg, **kwargs):
        super().__init__(**kwargs)
        self.tta_cfg = tta_cfg

    def merge_preds(self, data_samples_list: List[List[DetDataSample]]):
        aug_bboxes = []
        aug_scores = []
        aug_labels = []
        aug_masks = []
        img_metas = []

        for data_samples in data_samples_list:
            _img_metas = []
            aug_bboxes.append(data_samples[0].pred_instances.bboxes)
            aug_scores.append(data_samples[0].pred_instances.scores)
            aug_labels.append(data_samples[0].pred_instances.labels)
            if data_samples[0].pred_instances.get('masks', None) is not None:
                aug_masks.append(data_samples[0].pred_instances.masks)
            _img_metas.append(data_samples[0].metainfo)
            img_metas.append(_img_metas)

        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_masks = merge_aug_masks(aug_masks,
                                       img_metas) if aug_masks else None
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
        det_masks = merged_masks[
            keep_idxs][:self.tta_cfg.
                       max_per_img] if merged_masks is not None else None

        results = InstanceData()
        _det_bboxes = det_bboxes.clone()
        results.bboxes = _det_bboxes[:, :4]
        results.scores = _det_bboxes[:, 4]
        results.labels = det_labels
        if len(aug_masks) > 0:
            results.masks = det_masks
        det_results = data_samples_list[0][0]
        det_results.pred_instances = results
        return [det_results]
