# Copyright (c) OpenMMLab. All rights reserved.
import torch


def masked_fill(ori_tensor, mask, new_value, neg=False):
    """The Value of ori_tensor is new_value, depending on mask.

    Args:
        ori_tensor (Tensor): Input tensor.
        mask (Tensor): If select new_value.
        new_value(Tensor | scalar): Value selected for ori_tensor.
        neg (bool): If True, select ori_tensor. If False, select new_value.
    Returns:
        ori_tensor: (Tensor): The Value of ori_tensor is new_value,
            depending on mask.
    """
    if mask is None:
        return ori_tensor
    else:
        if neg:
            return ori_tensor * mask + new_value * (1 - mask)
        else:
            return ori_tensor * (1 - mask) + new_value * mask


def batch_images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]  or
    target_imgs -> [target_level0, target_level1, ...]
    Args:
        target (Tensor | List[Tensor]): Tensor split to image levels.
        num_levels (List[int]): Image levels num.
    Returns:
        level_targets: (Tensor): Tensor split by image levels.
    """
    if not isinstance(target, torch.Tensor):
        target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def get_max_num_gt_division_factor(gt_nums,
                                   min_num_gt=32,
                                   max_num_gt=1024,
                                   division_factor=2):
    """Count max num of gt.

    Args:
        gt_nums (List[int]):  Ground truth bboxes num of images.
        min_num_gt (int): Min num of ground truth bboxes.
        max_num_gt (int): Max num of ground truth bboxes.
        division_factor (int): Division factor of result.
    Returns:
        max_gt_nums_align: (int): max num of ground truth bboxes.
    """
    max_gt_nums = max(gt_nums)
    max_gt_nums_align = min_num_gt
    while max_gt_nums_align < max_gt_nums:
        max_gt_nums_align *= division_factor
    if max_gt_nums_align > max_num_gt:
        raise RuntimeError
    return max_gt_nums_align
