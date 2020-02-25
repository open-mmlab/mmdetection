import copy

import mmcv
import numpy as np
import pycocotools.mask as mask_utils
import torch
from torch.nn.modules.utils import _pair


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg, img_meta_list):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    if isinstance(gt_masks_list[0], np.ndarray):
        mask_target_single = mask_target_single_bitmaps
    elif isinstance(gt_masks_list[0], list):
        mask_target_single = mask_target_single_polygons
    else:
        raise NotImplementedError
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list,
                       img_meta_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


def mask_target_single_bitmaps(pos_proposals, pos_assigned_gt_inds, gt_masks,
                               cfg, img_meta):
    mask_size = _pair(cfg.mask_size)
    num_pos = pos_proposals.size(0)
    mask_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        _, maxh, maxw = gt_masks.shape
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw - 1)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh - 1)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            # mask is uint8 both before and after resizing
            # mask_size (h, w) to (w, h)
            target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                   mask_size[::-1])
            mask_targets.append(target)
        mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
            pos_proposals.device)
    else:
        mask_targets = pos_proposals.new_zeros((0, ) + mask_size)
    return mask_targets


def mask_target_single_polygons(pos_proposals, pos_assigned_gt_inds, gt_masks,
                                cfg, img_meta):
    mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    mask_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        maxh, maxw = img_meta['pad_shape'][:2]
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw - 1)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh - 1)
        for i in range(num_pos):
            polygons = gt_masks[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)

            # shift
            polygons = copy.deepcopy(polygons)
            for p in polygons:
                p[0::2] = p[0::2] - bbox[0]
                p[1::2] = p[1::2] - bbox[1]

            # rescale
            w = bbox[2] - bbox[0] + 1
            h = bbox[3] - bbox[1] + 1
            ratio_h = mask_size / h
            ratio_w = mask_size / w
            if ratio_h == ratio_w:
                for p in polygons:
                    p *= ratio_h
            else:
                for p in polygons:
                    p[0::2] *= ratio_w
                    p[1::2] *= ratio_h

            # convert to bitmap
            rles = mask_utils.frPyObjects(polygons, mask_size, mask_size)
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle).astype(np.bool)
            mask_targets.append(torch.from_numpy(mask))
        mask_targets = torch.stack(mask_targets).to(
            device=pos_proposals.device).float()
    else:
        mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
    return mask_targets
