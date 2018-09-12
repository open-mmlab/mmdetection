import torch
import numpy as np

from .segms import polys_to_mask_wrt_box


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_polys_list,
                img_meta, cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    img_metas = [img_meta for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_polys_list, img_metas,
                       cfg_list)
    mask_targets = torch.cat(tuple(mask_targets), dim=0)
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_polys,
                       img_meta, cfg):

    mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    mask_targets = pos_proposals.new_zeros((num_pos, mask_size, mask_size))
    if num_pos > 0:
        pos_proposals = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        scale_factor = img_meta['scale_factor'][0].cpu().numpy()
        for i in range(num_pos):
            bbox = pos_proposals[i, :] / scale_factor
            polys = gt_polys[pos_assigned_gt_inds[i]]
            mask = polys_to_mask_wrt_box(polys, bbox, mask_size)
            mask = np.array(mask > 0, dtype=np.float32)
            mask_targets[i, ...] = torch.from_numpy(mask).to(
                mask_targets.device)
    return mask_targets
