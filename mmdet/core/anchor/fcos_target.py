import torch

from ..utils import multi_apply

INF = 1e8


def fcos_target(centers, regress_ranges, gt_bboxes_list, gt_labels_list):
    assert len(centers) == len(regress_ranges)
    num_levels = len(centers)
    # expand regress ranges to align with centers
    expanded_regress_ranges = [
        centers[i].new_tensor(regress_ranges[i])[None].expand_as(centers[i])
        for i in range(num_levels)
    ]
    # concat all levels centers and regress ranges
    concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
    concat_centers = torch.cat(centers, dim=0)
    # get labels and bbox_targets of each image
    labels_list, bbox_targets_list = multi_apply(
        fcos_target_single,
        gt_bboxes_list,
        gt_labels_list,
        centers=concat_centers,
        regress_ranges=concat_regress_ranges)

    # split to per img, per level
    num_centers = [center.size(0) for center in centers]
    labels_list = [labels.split(num_centers, 0) for labels in labels_list]
    bbox_targets_list = [
        bbox_targets.split(num_centers, 0)
        for bbox_targets in bbox_targets_list
    ]

    # concat per level image
    concat_lvl_labels = []
    concat_lvl_bbox_targets = []
    for i in range(num_levels):
        concat_lvl_labels.append(
            torch.cat([labels[i] for labels in labels_list]))
        concat_lvl_bbox_targets.append(
            torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list]))
    return concat_lvl_labels, concat_lvl_bbox_targets


def fcos_target_single(gt_bboxes, gt_labels, centers, regress_ranges):
    num_centers = centers.size(0)
    num_gts = gt_labels.size(0)

    areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
        gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
    # TODO: figure out why these two different
    # areas = areas[None].expand(num_centers, num_gts)
    areas = areas[None].repeat(num_centers, 1)
    regress_ranges = regress_ranges[:, None, :].expand(num_centers, num_gts, 2)
    gt_bboxes = gt_bboxes[None].expand(num_centers, num_gts, 4)
    xs, ys = centers[:, 0], centers[:, 1]
    xs = xs[:, None].expand(num_centers, num_gts)
    ys = ys[:, None].expand(num_centers, num_gts)

    left = xs - gt_bboxes[..., 0]
    right = gt_bboxes[..., 2] - xs
    top = ys - gt_bboxes[..., 1]
    bottom = gt_bboxes[..., 3] - ys
    bbox_targets = torch.stack((left, top, right, bottom), -1)

    # condition1: inside a gt bbox
    inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

    # condition2: regress limited to the regress range
    max_regress_distance = bbox_targets.max(-1)[0]
    inside_regress_range = (max_regress_distance >= regress_ranges[..., 0]) & (
        max_regress_distance <= regress_ranges[..., 1])

    # condition3: if one center inside multi gts, choose smallest area gt
    areas[inside_gt_bbox_mask == 0] = INF
    areas[inside_regress_range == 0] = INF
    min_area, min_area_inds = areas.min(dim=1)

    labels = gt_labels[min_area_inds]
    labels[min_area == INF] = 0
    bbox_targets = bbox_targets[range(num_centers), min_area_inds]

    return labels, bbox_targets


def centerness_target(pos_bbox_targets):
    # only calculate pos centerness targets, otherwise there may be nan
    left_right = pos_bbox_targets[:, [0, 2]]
    top_bottom = pos_bbox_targets[:, [1, 3]]
    centerness_targets = (left_right.min(dim=-1)[0] / left_right.max(
        dim=-1)[0]) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness_targets)
