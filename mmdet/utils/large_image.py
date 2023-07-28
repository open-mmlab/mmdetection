# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple

import torch
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData

from mmdet.structures import DetDataSample, SampleList


def shift_rbboxes(bboxes: torch.Tensor, offset: Sequence[int]):
    """Shift rotated bboxes with offset.

    Args:
        bboxes (Tensor): The rotated bboxes need to be translated.
            With shape (n, 5), which means (x, y, w, h, a).
        offset (Sequence[int]): The translation offsets with shape of (2, ).
    Returns:
        Tensor: Shifted rotated bboxes.
    """
    offset_tensor = bboxes.new_tensor(offset)
    shifted_bboxes = bboxes.clone()
    shifted_bboxes[:, 0:2] = shifted_bboxes[:, 0:2] + offset_tensor
    return shifted_bboxes


def shift_predictions(det_data_samples: SampleList,
                      offsets: Sequence[Tuple[int, int]],
                      src_image_shape: Tuple[int, int]) -> SampleList:
    """Shift predictions to the original image.

    Args:
        det_data_samples (List[:obj:`DetDataSample`]): A list of patch results.
        offsets (Sequence[Tuple[int, int]]): Positions of the left top points
            of patches.
        src_image_shape (Tuple[int, int]): A (height, width) tuple of the large
            image's width and height.
    Returns:
        (List[:obj:`DetDataSample`]): shifted results.
    """
    try:
        from sahi.slicing import shift_bboxes, shift_masks
    except ImportError:
        raise ImportError('Please run "pip install -U sahi" '
                          'to install sahi first for large image inference.')

    assert len(det_data_samples) == len(
        offsets), 'The `results` should has the ' 'same length with `offsets`.'
    shifted_predictions = []
    for det_data_sample, offset in zip(det_data_samples, offsets):
        pred_inst = det_data_sample.pred_instances.clone()

        # Check bbox type
        if pred_inst.bboxes.size(-1) == 4:
            # Horizontal bboxes
            shifted_bboxes = shift_bboxes(pred_inst.bboxes, offset)
        elif pred_inst.bboxes.size(-1) == 5:
            # Rotated bboxes
            shifted_bboxes = shift_rbboxes(pred_inst.bboxes, offset)
        else:
            raise NotImplementedError

        # shift bboxes and masks
        pred_inst.bboxes = shifted_bboxes
        if 'masks' in det_data_sample:
            pred_inst.masks = shift_masks(pred_inst.masks, offset,
                                          src_image_shape)

        shifted_predictions.append(pred_inst.clone())

    shifted_predictions = InstanceData.cat(shifted_predictions)

    return shifted_predictions


def merge_results_by_nms(results: SampleList, offsets: Sequence[Tuple[int,
                                                                      int]],
                         src_image_shape: Tuple[int, int],
                         nms_cfg: dict) -> DetDataSample:
    """Merge patch results by nms.

    Args:
        results (List[:obj:`DetDataSample`]): A list of patch results.
        offsets (Sequence[Tuple[int, int]]): Positions of the left top points
            of patches.
        src_image_shape (Tuple[int, int]): A (height, width) tuple of the large
            image's width and height.
        nms_cfg (dict): it should specify nms type and other parameters
            like `iou_threshold`.
    Returns:
        :obj:`DetDataSample`: merged results.
    """
    shifted_instances = shift_predictions(results, offsets, src_image_shape)

    _, keeps = batched_nms(
        boxes=shifted_instances.bboxes,
        scores=shifted_instances.scores,
        idxs=shifted_instances.labels,
        nms_cfg=nms_cfg)
    merged_instances = shifted_instances[keeps]

    merged_result = results[0].clone()
    merged_result.pred_instances = merged_instances
    return merged_result
