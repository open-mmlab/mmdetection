import copy

import numpy as np
import pycocotools.mask as mask_utils
import torch
from torch.nn.modules.utils import _pair

from mmdet.ops import roi_align


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    if isinstance(gt_masks_list[0], np.ndarray):
        mask_target_single = mask_target_single_bitmaps
    elif isinstance(gt_masks_list[0], list):
        mask_target_single = mask_target_single_polygons
    else:
        raise NotImplementedError
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


def mask_target_single_polygons(pos_proposals, pos_assigned_gt_inds, gt_masks,
                                cfg):
    mask_size = cfg.mask_size
    boxes = pos_proposals.to(torch.device('cpu')).numpy()
    num_pos = pos_proposals.size(0)

    if num_pos > 0:
        gt_masks = [gt_masks[i] for i in pos_assigned_gt_inds]
        mask_targets = []
        for polygons, box in zip(gt_masks, boxes):
            # 1. Shift the polygons w.r.t the boxes
            polygons = copy.deepcopy(polygons)
            for p in polygons:
                p[0::2] = p[0::2] - box[0]
                p[1::2] = p[1::2] - box[1]

            # 2. Rescale the polygons to the new box size
            w, h = box[2] - box[0], box[3] - box[1]
            ratio_h = mask_size / max(h, 0.1)
            ratio_w = mask_size / max(w, 0.1)
            if ratio_h == ratio_w:
                for p in polygons:
                    p *= ratio_h
            else:
                for p in polygons:
                    p[0::2] *= ratio_w
                    p[1::2] *= ratio_h

            # 3. Rasterize the polygons with coco api
            rles = mask_utils.frPyObjects(polygons, mask_size, mask_size)
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle).astype(np.bool)
            mask_targets.append(torch.from_numpy(mask))
        mask_targets = torch.stack(mask_targets).to(
            device=pos_proposals.device).float()
    else:
        mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
    return mask_targets


def mask_target_single_bitmaps(pos_proposals, pos_assigned_gt_inds, gt_masks,
                               cfg):
    device = pos_proposals.device
    mask_size = _pair(cfg.mask_size)
    num_pos = pos_proposals.size(0)
    fake_inds = (
        torch.arange(num_pos,
                     device=device).to(dtype=pos_proposals.dtype)[:, None])
    rois = torch.cat([fake_inds, pos_proposals], dim=1)  # Nx5
    rois = rois.to(device=device)
    if num_pos > 0:
        gt_masks_th = (
            torch.from_numpy(gt_masks).to(device).index_select(
                0, pos_assigned_gt_inds).to(dtype=rois.dtype))
        # Use RoIAlign could apparently accelerate the training (~0.1s/iter)
        targets = (
            roi_align(gt_masks_th[:, None, :, :], rois, mask_size[::-1], 1.0,
                      0, True).squeeze(1))
        # It is important to set the target > threshold rather
        # than >= (~0.5mAP)
        mask_targets = (targets >= 0.5).float()
    else:
        mask_targets = pos_proposals.new_zeros((0, ) + mask_size)
    return mask_targets
