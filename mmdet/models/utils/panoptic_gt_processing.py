# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch import Tensor


def preprocess_panoptic_gt(
        gt_labels: Tensor,
        gt_masks: Tensor,
        gt_semantic_seg: Tensor,
        num_things: int,
        num_stuff: int,
        merge_things_stuff: bool = True) -> Tuple[Tensor, Tensor]:
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
        merge_things_stuff (bool): Whether merges ground truth of things and
            ground truth of stuff together. Defaults to True.

    Returns:
        tuple: According to the value of the parameter, the return value
        can be divided into the following four cases:
        1. (things_labels, things_masks, None, None), when ``gt_semantic_seg``
        is None and ``merge_things_stuff`` is True.
        2. (labels, masks, None, None), when ``gt_semantic_seg`` is Not None
        and ``merge_things_stuff`` is True. The ``labels`` contains labels of
        things and labels of stuff, and so does the ``masks``.
        3. (things_labels, things_masks, None, None), when ``gt_semantic_seg``
        is None and ``merge_things_stuff`` is False.
        4. (things_labels, things_masks, stuff_labels, stuff_masks), when
        ``gt_semantic_seg`` is Not None and ``merge_things_stuff`` is False.

        The shape of labels(things_labels and stuff_labels) is like (n, ),
        and the shape of masks(things_masks and stuff_masks) is like (n, h, w).
    """
    num_classes = num_things + num_stuff

    things_masks = gt_masks.to_tensor(
        dtype=torch.bool, device=gt_labels.device)
    things_labels = gt_labels

    if gt_semantic_seg is None:
        return things_labels, things_masks.long(), None, None

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

    if not merge_things_stuff:
        if len(stuff_labels_list) > 0:
            stuff_masks = torch.stack(stuff_masks_list, dim=0)
            stuff_labels = torch.stack(stuff_labels_list, dim=0).long()
        else:
            stuff_masks = gt_semantic_seg.new_zeros(
                size=(0, ) + gt_semantic_seg.shape[-2:])
            stuff_labels = gt_semantic_seg.new_zeros(size=(0, ))

        return things_labels, things_masks.long(), stuff_labels, stuff_masks
    else:
        if len(stuff_masks_list) > 0:
            stuff_masks = torch.stack(stuff_masks_list, dim=0)
            stuff_labels = torch.stack(stuff_labels_list, dim=0).long()
            labels = torch.cat([things_labels, stuff_labels], dim=0)
            masks = torch.cat([things_masks, stuff_masks], dim=0)
        else:
            labels = things_labels
            masks = things_masks

        return labels, masks.long(), None, None
