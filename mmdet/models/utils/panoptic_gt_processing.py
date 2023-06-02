# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch import Tensor


def preprocess_panoptic_gt(gt_labels: Tensor, gt_masks: Tensor,
                           gt_semantic_seg: Tensor, num_things: int,
                           num_stuff: int) -> Tuple[Tensor, Tensor]:
    """Preprocess the ground truth for a image.

    Args:
        gt_labels (Tensor): Ground truth labels of each bbox,
            with shape (num_gts, ).
        gt_masks (BitmapMasks): Ground truth masks of each instances
            of a image, shape (num_gts, h, w).
        gt_semantic_seg (Tensor | None): Ground truth of semantic
            segmentation with the shape (1, h, w).
            [0, num_thing_class - 1] means things,
            [num_thing_class, num_class-1] means stuff,
            255 means VOID. It's None when training instance segmentation.

    Returns:
        tuple[Tensor, Tensor]: a tuple containing the following targets.

            - labels (Tensor): Ground truth class indices for a
                image, with shape (n, ), n is the sum of number
                of stuff type and number of instance in a image.
            - masks (Tensor): Ground truth mask for a image, with
                shape (n, h, w). Contains stuff and things when training
                panoptic segmentation, and things only when training
                instance segmentation.
    """
    num_classes = num_things + num_stuff
    things_masks = gt_masks.to_tensor(
        dtype=torch.bool, device=gt_labels.device)

    if gt_semantic_seg is None:
        masks = things_masks.long()
        return gt_labels, masks

    things_labels = gt_labels
    gt_semantic_seg = gt_semantic_seg.squeeze(0)

    semantic_labels = torch.unique(
        gt_semantic_seg,
        sorted=False,
        return_inverse=False,
        return_counts=False)
    stuff_masks_list = []
    stuff_labels_list = []
    for label in semantic_labels:
        if label < num_things or label >= num_classes:
            continue
        stuff_mask = gt_semantic_seg == label
        stuff_masks_list.append(stuff_mask)
        stuff_labels_list.append(label)

    if len(stuff_masks_list) > 0:
        stuff_masks = torch.stack(stuff_masks_list, dim=0)
        stuff_labels = torch.stack(stuff_labels_list, dim=0)
        labels = torch.cat([things_labels, stuff_labels], dim=0)
        masks = torch.cat([things_masks, stuff_masks], dim=0)
    else:
        labels = things_labels
        masks = things_masks

    masks = masks.long()
    return labels, masks
