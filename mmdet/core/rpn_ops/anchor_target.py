import torch
import numpy as np
from ..bbox_ops import (bbox_assign, bbox_transform, bbox_sampling)


def anchor_target(anchor_list, valid_flag_list, featmap_sizes, gt_bboxes_list,
                  img_shapes, target_means, target_stds, cfg):
    """Compute anchor regression and classification targets

    Args:
        anchor_list(list): anchors of each feature map level
        featuremap_sizes(list): feature map sizes
        gt_bboxes_list(list): ground truth bbox of images in a mini-batch
        img_shapes(list): shape of each image in a mini-batch
        cfg(dict): configs

    Returns:
        tuple
    """
    if len(featmap_sizes) == len(anchor_list):
        all_anchors = torch.cat(anchor_list, 0)
        anchor_nums = [anchors.size(0) for anchors in anchor_list]
        use_isomerism_anchors = False
    elif len(img_shapes) == len(anchor_list):
        # using different anchors for different images
        all_anchors_list = [
            torch.cat(anchor_list[img_id], 0)
            for img_id in range(len(img_shapes))
        ]
        anchor_nums = [anchors.size(0) for anchors in anchor_list[0]]
        use_isomerism_anchors = True
    else:
        raise ValueError('length of anchor_list should be equal to number of '
                         'feature lvls or number of images in a batch')
    all_labels = []
    all_label_weights = []
    all_bbox_targets = []
    all_bbox_weights = []
    num_total_sampled = 0
    for img_id in range(len(img_shapes)):
        if isinstance(valid_flag_list[img_id], list):
            valid_flags = torch.cat(valid_flag_list[img_id], 0)
        else:
            valid_flags = valid_flag_list[img_id]
        if use_isomerism_anchors:
            all_anchors = all_anchors_list[img_id]
        inside_flags = anchor_inside_flags(all_anchors, valid_flags,
                                           img_shapes[img_id][:2],
                                           cfg.allowed_border)
        if not inside_flags.any():
            return None
        gt_bboxes = gt_bboxes_list[img_id]
        anchor_targets = anchor_target_single(all_anchors, inside_flags,
                                              gt_bboxes, target_means,
                                              target_stds, cfg)
        (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
         neg_inds) = anchor_targets
        all_labels.append(labels)
        all_label_weights.append(label_weights)
        all_bbox_targets.append(bbox_targets)
        all_bbox_weights.append(bbox_weights)
        num_total_sampled += max(pos_inds.numel() + neg_inds.numel(), 1)
    all_labels = torch.stack(all_labels, 0)
    all_label_weights = torch.stack(all_label_weights, 0)
    all_bbox_targets = torch.stack(all_bbox_targets, 0)
    all_bbox_weights = torch.stack(all_bbox_weights, 0)
    # split into different feature levels
    labels_list = []
    label_weights_list = []
    bbox_targets_list = []
    bbox_weights_list = []
    start = 0
    for anchor_num in anchor_nums:
        end = start + anchor_num
        labels_list.append(all_labels[:, start:end].squeeze(0))
        label_weights_list.append(all_label_weights[:, start:end].squeeze(0))
        bbox_targets_list.append(all_bbox_targets[:, start:end].squeeze(0))
        bbox_weights_list.append(all_bbox_weights[:, start:end].squeeze(0))
        start = end
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_sampled)


def anchor_target_single(all_anchors, inside_flags, gt_bboxes, target_means,
                         target_stds, cfg):
    num_total_anchors = all_anchors.size(0)
    anchors = all_anchors[inside_flags, :]
    assigned_gt_inds, argmax_overlaps, max_overlaps = bbox_assign(
        anchors,
        gt_bboxes,
        pos_iou_thr=cfg.pos_iou_thr,
        neg_iou_thr=cfg.neg_iou_thr,
        min_pos_iou=cfg.min_pos_iou)
    pos_inds, neg_inds = bbox_sampling(assigned_gt_inds, cfg.anchor_batch_size,
                                       cfg.pos_fraction, cfg.neg_pos_ub,
                                       cfg.pos_balance_sampling, max_overlaps,
                                       cfg.neg_balance_thr)

    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = torch.zeros_like(assigned_gt_inds)
    label_weights = torch.zeros_like(assigned_gt_inds, dtype=torch.float)

    if len(pos_inds) > 0:
        pos_inds = unique(pos_inds)
        pos_anchors = anchors[pos_inds, :]
        pos_gt_bbox = gt_bboxes[assigned_gt_inds[pos_inds] - 1, :]
        pos_bbox_targets = bbox_transform(pos_anchors, pos_gt_bbox,
                                          target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        labels[pos_inds] = 1
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        neg_inds = unique(neg_inds)
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    labels = unmap(labels, num_total_anchors, inside_flags)
    label_weights = unmap(label_weights, num_total_anchors, inside_flags)
    bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
    bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)

def anchor_inside_flags(all_anchors, valid_flags, img_shape, allowed_border=0):
    img_h, img_w = img_shape.float()
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (all_anchors[:, 0] >= -allowed_border) & \
            (all_anchors[:, 1] >= -allowed_border) & \
            (all_anchors[:, 2] < img_w + allowed_border) & \
            (all_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags

def unique(tensor):
    if tensor.is_cuda:
        u_tensor = np.unique(tensor.cpu().numpy())
        return tensor.new_tensor(u_tensor)
    else:
        return torch.unique(tensor)

def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
