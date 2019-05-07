import mmcv
import numpy as np
import torch

from ..utils import multi_apply


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets, area_ratios = multi_apply(
        mask_target_single, pos_proposals_list, pos_assigned_gt_inds_list,
        gt_masks_list, cfg_list)
    mask_targets = torch.cat(mask_targets)
    area_ratios = torch.cat(area_ratios)
    return mask_targets, area_ratios


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    mask_targets = []
    area_ratios = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            # mask is uint8 both before and after resizing
            target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                   (mask_size, mask_size))
            ratio = gt_mask[y1:y1 + h, x1:x1 + w].sum() / gt_mask.sum()
            mask_targets.append(target)
            area_ratios.append(ratio)
        mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
            pos_proposals.device)
        area_ratios = torch.from_numpy(np.stack(area_ratios)).float().to(
            pos_proposals.device)
    else:
        mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
        area_ratios = pos_proposals.new_zeros((0, ))
    return mask_targets, area_ratios
