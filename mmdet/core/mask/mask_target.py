import mmcv
import numpy as np
import torch


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    mask_targets = []
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
            mask_targets.append(target)
        mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
            pos_proposals.device)
    else:
        mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
    return mask_targets


def mask_iou_target(pos_proposals_list, pos_assigned_gt_inds_list,
                    gt_masks_list, mask_preds, mask_targets, cfg):
    """Compute mask IoU target of each positive proposal.

    Mask IoU target is the IoU of the predicted mask and the gt mask of
    corresponding gt instance. GT masks are in image space, so firstly compute
    the area ratio (please refer to get_area_ratio func) and use it to compute
    the full area of the instance at the scale of mask pred.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals of each image,
            shape of each pos proposals (n, 4).
        pos_assigned_gt_inds_list (list[Tensor]): Positive assigned gt inds of
            each image, shape of each pos_assigned_gt_inds (n,).
        gt_masks_list (list[ndarray]): Ground truth mask of each image, shape
            of gt_masks (num_obj, img height, img width)
        mask_preds (Tensor): Predicted masks of each positive proposal,
            shape (num_pos, h, w)
        mask_targets (Tensor): Target mask of each positive proposal,
            shape (num_pos, h, w)

    Returns:
        Tensor: mask iou target (length == num positive)
    """
    area_ratios = map(get_area_ratio, pos_proposals_list,
                      pos_assigned_gt_inds_list, gt_masks_list)
    area_ratios = torch.cat(list(area_ratios))
    assert mask_targets.size(0) == area_ratios.size(0)

    mask_pred = (mask_preds > cfg.mask_thr_binary).float()

    # mask target is also either 0 or 1
    mask_overlaps = (mask_pred * mask_targets).sum((-1, -2))

    # mask area of the whole instance
    full_areas = mask_targets.sum((-1, -2)) / area_ratios

    mask_unions = mask_pred.sum((-1, -2)) + full_areas - mask_overlaps

    mask_iou_targets = mask_overlaps / mask_unions
    return mask_iou_targets


def get_area_ratio(pos_proposals, pos_assigned_gt_inds, gt_masks):
    """Compute area ratio of the gt mask inside the proposal and the gt mask
    of the corresponding instance"""
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        area_ratios = []
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        # only sum gt mask once to save time
        gt_instance_mask_area = gt_masks.sum((-1, -2))
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]

            x1, y1, x2, y2 = proposals_np[i, :].astype(np.int32)
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)

            # crop the gt mask inside the proposal
            gt_mask_in_proposal = gt_mask[y1:y1 + h, x1:x1 + w]

            ratio = gt_mask_in_proposal.sum() / (
                gt_instance_mask_area[pos_assigned_gt_inds[i]] + 1e-7
            )  # avoid zero
            area_ratios.append(ratio)
        area_ratios = torch.from_numpy(np.stack(area_ratios)).float().to(
            pos_proposals.device) + 1e-7  # avoid zero
    else:
        area_ratios = pos_proposals.new_zeros((0, ))
    return area_ratios
