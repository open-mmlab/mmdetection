import mmcv
import torch

from .geometry import bbox_overlaps
from .transforms import bbox_transform, bbox_transform_inv


def bbox_target(pos_proposals_list,
                neg_proposals_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                cfg,
                reg_num_classes=1,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                return_list=False):
    img_per_gpu = len(pos_proposals_list)
    all_labels = []
    all_label_weights = []
    all_bbox_targets = []
    all_bbox_weights = []
    for img_id in range(img_per_gpu):
        pos_proposals = pos_proposals_list[img_id]
        neg_proposals = neg_proposals_list[img_id]
        pos_gt_bboxes = pos_gt_bboxes_list[img_id]
        pos_gt_labels = pos_gt_labels_list[img_id]
        debug_img = debug_imgs[img_id] if cfg.debug else None
        labels, label_weights, bbox_targets, bbox_weights = proposal_target_single(
            pos_proposals, neg_proposals, pos_gt_bboxes, pos_gt_labels,
            reg_num_classes, cfg, target_means, target_stds)
        all_labels.append(labels)
        all_label_weights.append(label_weights)
        all_bbox_targets.append(bbox_targets)
        all_bbox_weights.append(bbox_weights)

    if return_list:
        return all_labels, all_label_weights, all_bbox_targets, all_bbox_weights

    labels = torch.cat(all_labels, 0)
    label_weights = torch.cat(all_label_weights, 0)
    bbox_targets = torch.cat(all_bbox_targets, 0)
    bbox_weights = torch.cat(all_bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


def proposal_target_single(pos_proposals,
                           neg_proposals,
                           pos_gt_bboxes,
                           pos_gt_labels,
                           reg_num_classes,
                           cfg,
                           target_means=[.0, .0, .0, .0],
                           target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_proposals.size(0)
    num_neg = neg_proposals.size(0)
    num_samples = num_pos + num_neg
    labels = pos_proposals.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_proposals.new_zeros(num_samples)
    bbox_targets = pos_proposals.new_zeros(num_samples, 4)
    bbox_weights = pos_proposals.new_zeros(num_samples, 4)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox_transform(pos_proposals, pos_gt_bboxes,
                                          target_means, target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0
    if reg_num_classes > 1:
        bbox_targets, bbox_weights = expand_target(bbox_targets, bbox_weights,
                                                   labels, reg_num_classes)

    return labels, label_weights, bbox_targets, bbox_weights


def expand_target(bbox_targets, bbox_weights, labels, num_classes):
    bbox_targets_expand = bbox_targets.new_zeros((bbox_targets.size(0),
                                                  4 * num_classes))
    bbox_weights_expand = bbox_weights.new_zeros((bbox_weights.size(0),
                                                  4 * num_classes))
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        bbox_weights_expand[i, start:end] = bbox_weights[i, :]
    return bbox_targets_expand, bbox_weights_expand
