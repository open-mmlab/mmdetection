import torch

from ..bbox import PseudoSampler, bbox2delta, build_assigner, build_sampler
from ..utils import multi_apply


def region_anchor_target(anchor_list,
                         valid_flag_list,
                         gt_bboxes_list,
                         img_metas,
                         featmap_sizes,
                         anchor_scale,
                         anchor_strides,
                         target_means,
                         target_stds,
                         cfg,
                         gt_bboxes_ignore_list=None,
                         gt_labels_list=None,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True):
    # TODO add docs
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

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         region_anchor_target_single,
         anchor_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         featmap_sizes=featmap_sizes,
         anchor_scale=anchor_scale,
         anchor_strides=anchor_strides,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    bbox_anchor_list = anchor2rois(anchor_list, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg, bbox_anchor_list)


def anchor2rois(anchor_list, num_level_anchors):
    _anchor_list = []
    num_imgs = len(anchor_list)
    for i in range(num_imgs):
        _anchor_list.append(torch.cat(anchor_list[i]))
    rois_list = images_to_levels(_anchor_list, num_level_anchors)
    return rois_list


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


def region_anchor_target_single(anchors,
                                valid_flags,
                                gt_bboxes,
                                gt_bboxes_ignore,
                                gt_labels,
                                img_meta,
                                featmap_sizes,
                                anchor_scale,
                                anchor_strides,
                                target_means,
                                target_stds,
                                cfg,
                                label_channels=1,  # TODO: check this arg
                                sampling=True):  # yapf: disable
    bbox_assigner = build_assigner(cfg.assigner)
    assign_result = bbox_assigner.assign(
        anchors,
        valid_flags,
        gt_bboxes,
        img_meta,
        featmap_sizes,
        anchor_scale,
        anchor_strides,
        gt_bboxes_ignore=gt_bboxes_ignore,
        gt_labels=None,
        allowed_border=cfg.allowed_border)
    bbox_sampler = build_sampler(cfg.sampler) if sampling else PseudoSampler()
    flat_anchors = torch.cat(anchors)
    sampling_result = bbox_sampler.sample(assign_result, flat_anchors,
                                          gt_bboxes)

    num_anchors = flat_anchors.shape[0]
    bbox_targets = torch.zeros_like(flat_anchors)
    bbox_weights = torch.zeros_like(flat_anchors)
    labels = flat_anchors.new_zeros(num_anchors, dtype=torch.long)
    label_weights = flat_anchors.new_zeros(num_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)
