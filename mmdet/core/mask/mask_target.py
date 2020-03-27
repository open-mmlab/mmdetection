import numpy as np
import torch
from torch.nn.modules.utils import _pair


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    mask_size = _pair(cfg.mask_size)
    num_pos = pos_proposals.size(0)
    mask_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            bbox = proposals_np[i].astype(np.int32)
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            cropped_mask = gt_mask.crop(bbox)
            resized_mask = cropped_mask.resize(
                h=mask_size[0], w=mask_size[1], interpolation='bilinear')
            mask_targets.append(
                resized_mask.to_tensor(torch.float,
                                       pos_proposals.device).squeeze())
        mask_targets = torch.stack(mask_targets)
    else:
        mask_targets = pos_proposals.new_zeros((0, ) + mask_size)
    return mask_targets
