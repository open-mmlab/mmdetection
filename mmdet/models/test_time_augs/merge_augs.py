# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Optional, Union

import numpy as np
import torch
from mmcv.ops import nms
from mmengine.config import ConfigDict
from torch import Tensor

from mmdet.structures.bbox import bbox_mapping_back


# TODO remove this, never be used in mmdet
def merge_aug_proposals(aug_proposals, img_metas, cfg):
    """Merge augmented proposals (multiscale, flip, etc.)

    Args:
        aug_proposals (list[Tensor]): proposals from different testing
            schemes, shape (n, 5). Note that they are not rescaled to the
            original image size.

        img_metas (list[dict]): list of image info dict where each dict has:
            'img_shape', 'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `mmdet/datasets/pipelines/formatting.py:Collect`.

        cfg (dict): rpn test config.

    Returns:
        Tensor: shape (n, 4), proposals corresponding to original image scale.
    """

    cfg = copy.deepcopy(cfg)

    # deprecate arguments warning
    if 'nms' not in cfg or 'max_num' in cfg or 'nms_thr' in cfg:
        warnings.warn(
            'In rpn_proposal or test_cfg, '
            'nms_thr has been moved to a dict named nms as '
            'iou_threshold, max_num has been renamed as max_per_img, '
            'name of original arguments and the way to specify '
            'iou_threshold of NMS will be deprecated.')
    if 'nms' not in cfg:
        cfg.nms = ConfigDict(dict(type='nms', iou_threshold=cfg.nms_thr))
    if 'max_num' in cfg:
        if 'max_per_img' in cfg:
            assert cfg.max_num == cfg.max_per_img, f'You set max_num and ' \
                f'max_per_img at the same time, but get {cfg.max_num} ' \
                f'and {cfg.max_per_img} respectively' \
                f'Please delete max_num which will be deprecated.'
        else:
            cfg.max_per_img = cfg.max_num
    if 'nms_thr' in cfg:
        assert cfg.nms.iou_threshold == cfg.nms_thr, f'You set ' \
            f'iou_threshold in nms and ' \
            f'nms_thr at the same time, but get ' \
            f'{cfg.nms.iou_threshold} and {cfg.nms_thr}' \
            f' respectively. Please delete the nms_thr ' \
            f'which will be deprecated.'

    recovered_proposals = []
    for proposals, img_info in zip(aug_proposals, img_metas):
        img_shape = img_info['img_shape']
        scale_factor = img_info['scale_factor']
        flip = img_info['flip']
        flip_direction = img_info['flip_direction']
        _proposals = proposals.clone()
        _proposals[:, :4] = bbox_mapping_back(_proposals[:, :4], img_shape,
                                              scale_factor, flip,
                                              flip_direction)
        recovered_proposals.append(_proposals)
    aug_proposals = torch.cat(recovered_proposals, dim=0)
    merged_proposals, _ = nms(aug_proposals[:, :4].contiguous(),
                              aug_proposals[:, -1].contiguous(),
                              cfg.nms.iou_threshold)
    scores = merged_proposals[:, 4]
    _, order = scores.sort(0, descending=True)
    num = min(cfg.max_per_img, merged_proposals.shape[0])
    order = order[:num]
    merged_proposals = merged_proposals[order, :]
    return merged_proposals


# TODO remove this, never be used in mmdet
def merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg):
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 4*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        flip_direction = img_info[0]['flip_direction']
        bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                   flip_direction)
        recovered_bboxes.append(bboxes)
    bboxes = torch.stack(recovered_bboxes).mean(dim=0)
    if aug_scores is None:
        return bboxes
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores


def merge_aug_results(aug_batch_results, aug_batch_img_metas):
    """Merge augmented detection results, only bboxes corresponding score under
    flipping and multi-scale resizing can be processed now.

    Args:
        aug_batch_results (list[list[[obj:`InstanceData`]]):
            Detection results of multiple images with
            different augmentations.
            The outer list indicate the augmentation . The inter
            list indicate the batch dimension.
            Each item usually contains the following keys.

            - scores (Tensor): Classification scores, in shape
              (num_instance,)
            - labels (Tensor): Labels of bboxes, in shape
              (num_instances,).
            - bboxes (Tensor): In shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        aug_batch_img_metas (list[list[dict]]): The outer list
            indicates test-time augs (multiscale, flip, etc.)
            and the inner list indicates
            images in a batch. Each dict in the list contains
            information of an image in the batch.

    Returns:
        batch_results (list[obj:`InstanceData`]): Same with
        the input `aug_results` except that all bboxes have
        been mapped to the original scale.
    """
    num_augs = len(aug_batch_results)
    num_imgs = len(aug_batch_results[0])

    batch_results = []
    aug_batch_results = copy.deepcopy(aug_batch_results)
    for img_id in range(num_imgs):
        aug_results = []
        for aug_id in range(num_augs):
            img_metas = aug_batch_img_metas[aug_id][img_id]
            results = aug_batch_results[aug_id][img_id]

            img_shape = img_metas['img_shape']
            scale_factor = img_metas['scale_factor']
            flip = img_metas['flip']
            flip_direction = img_metas['flip_direction']
            bboxes = bbox_mapping_back(results.bboxes, img_shape, scale_factor,
                                       flip, flip_direction)
            results.bboxes = bboxes
            aug_results.append(results)
        merged_aug_results = results.cat(aug_results)
        batch_results.append(merged_aug_results)

    return batch_results


def merge_aug_scores(aug_scores):
    """Merge augmented bbox scores."""
    if isinstance(aug_scores[0], torch.Tensor):
        return torch.mean(torch.stack(aug_scores), dim=0)
    else:
        return np.mean(aug_scores, axis=0)


def merge_aug_masks(aug_masks: List[Tensor],
                    img_metas: dict,
                    weights: Optional[Union[list, Tensor]] = None) -> Tensor:
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
    for i, mask in enumerate(aug_masks):
        if weights is not None:
            assert len(weights) == len(aug_masks)
            weight = weights[i]
        else:
            weight = 1
        flip = img_metas.get('filp', False)
        if flip:
            flip_direction = img_metas['flip_direction']
            if flip_direction == 'horizontal':
                mask = mask[:, :, :, ::-1]
            elif flip_direction == 'vertical':
                mask = mask[:, :, ::-1, :]
            elif flip_direction == 'diagonal':
                mask = mask[:, :, :, ::-1]
                mask = mask[:, :, ::-1, :]
            else:
                raise ValueError(
                    f"Invalid flipping direction '{flip_direction}'")
        recovered_masks.append(mask[None, :] * weight)

    merged_masks = torch.cat(recovered_masks, 0).mean(dim=0)
    if weights is not None:
        merged_masks = merged_masks * len(weights) / sum(weights)
    return merged_masks
