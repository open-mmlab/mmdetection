# Copyright (c) OpenMMLab. All rights reserved.
import torch


def set_index(ori_tensor, mask, new_value, neg=False):
    if mask is None:
        return ori_tensor
    else:
        if neg:
            return ori_tensor * mask + new_value * (1 - mask)
        else:
            return ori_tensor * (1 - mask) + new_value * mask


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
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


def generate_max_gt_nums(gt_nums, minimum_gt_nums=32, maximum_gt_nums=1024):
    max_gt_nums = max(gt_nums)
    max_gt_nums_align = minimum_gt_nums
    while max_gt_nums_align < max_gt_nums:
        max_gt_nums_align *= 2
    if max_gt_nums_align > maximum_gt_nums:
        raise RuntimeError
    return max_gt_nums_align
