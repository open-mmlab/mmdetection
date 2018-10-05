import torch

from ..bbox_ops import bbox_assign, bbox2delta
from ..utils import multi_apply


def retina_target(anchor_list, valid_flag_list, gt_bboxes_list, gt_labels_list,
                  img_metas, target_means, target_stds, cls_out_channels, cfg):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         retina_target_single,
         anchor_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         cls_out_channels=cls_out_channels,
         cfg=cfg)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_pos_samples = sum([
        max(pos_inds.numel(), 1)
        for pos_inds, neg_inds in zip(pos_inds_list, neg_inds_list)
    ])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_pos_samples)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def retina_target_single(flat_anchors, valid_flags, gt_bboxes, gt_labels,
                         img_meta, target_means, target_stds, cls_out_channels,
                         cfg):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]
    assigned_gt_inds, argmax_overlaps, max_overlaps = bbox_assign(
        anchors,
        gt_bboxes,
        pos_iou_thr=cfg.pos_iou_thr,
        neg_iou_thr=cfg.neg_iou_thr,
        min_pos_iou=cfg.min_pos_iou)
    pos_inds = torch.nonzero(assigned_gt_inds > 0)
    neg_inds = torch.nonzero(assigned_gt_inds == 0)

    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = torch.zeros_like(assigned_gt_inds)
    label_weights = torch.zeros_like(assigned_gt_inds, dtype=anchors.dtype)

    if len(pos_inds) > 0:
        pos_inds = pos_inds.squeeze(1).unique()
        pos_anchors = anchors[pos_inds, :]
        pos_gt_bbox = gt_bboxes[assigned_gt_inds[pos_inds] - 1, :]
        pos_bbox_targets = bbox2delta(pos_anchors, pos_gt_bbox, target_means,
                                      target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        neg_inds = neg_inds.squeeze(1).unique()
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    num_total_anchors = flat_anchors.size(0)
    labels = unmap(labels, num_total_anchors, inside_flags)
    label_weights = unmap(label_weights, num_total_anchors, inside_flags)
    labels, label_weights = expand_binary_labels(labels, label_weights,
                                                 cls_out_channels)
    bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
    bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


def expand_binary_labels(labels, label_weights, cls_out_channels):
    bin_labels = labels.new_full(
        (labels.size(0), cls_out_channels), 0, dtype=torch.float32)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), cls_out_channels)
    return bin_labels, bin_label_weights


def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


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
