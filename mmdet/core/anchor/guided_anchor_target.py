import torch

from ..bbox import PseudoSampler, build_assigner, build_sampler
from ..utils import multi_apply, unmap


def calc_region(bbox, ratio, featmap_size=None):
    """Calculate a proportional bbox region.

    The bbox center are fixed and the new h' and w' is h * ratio and w * ratio.

    Args:
        bbox (Tensor): Bboxes to calculate regions, shape (n, 4)
        ratio (float): Ratio of the output region.
        featmap_size (tuple): Feature map size used for clipping the boundary.

    Returns:
        tuple: x1, y1, x2, y2
    """
    x1 = torch.round((1 - ratio) * bbox[0] + ratio * bbox[2]).long()
    y1 = torch.round((1 - ratio) * bbox[1] + ratio * bbox[3]).long()
    x2 = torch.round(ratio * bbox[0] + (1 - ratio) * bbox[2]).long()
    y2 = torch.round(ratio * bbox[1] + (1 - ratio) * bbox[3]).long()
    if featmap_size is not None:
        x1 = x1.clamp(min=0, max=featmap_size[1] - 1)
        y1 = y1.clamp(min=0, max=featmap_size[0] - 1)
        x2 = x2.clamp(min=0, max=featmap_size[1] - 1)
        y2 = y2.clamp(min=0, max=featmap_size[0] - 1)
    return (x1, y1, x2, y2)


def ga_loc_target(gt_bboxes_list,
                  featmap_sizes,
                  anchor_scale,
                  anchor_strides,
                  center_ratio=0.2,
                  ignore_ratio=0.5):
    """Compute location targets for guided anchoring.

    Each feature map is divided into positive, negative and ignore regions.
    - positive regions: target 1, weight 1
    - ignore regions: target 0, weight 0
    - negative regions: target 0, weight 0.1

    Args:
        gt_bboxes_list (list[Tensor]): Gt bboxes of each image.
        featmap_sizes (list[tuple]): Multi level sizes of each feature maps.
        anchor_scale (int): Anchor scale.
        anchor_strides ([list[int]]): Multi level anchor strides.
        center_ratio (float): Ratio of center region.
        ignore_ratio (float): Ratio of ignore region.

    Returns:
        tuple
    """
    img_per_gpu = len(gt_bboxes_list)
    num_lvls = len(featmap_sizes)
    r1 = (1 - center_ratio) / 2
    r2 = (1 - ignore_ratio) / 2
    all_loc_targets = []
    all_loc_weights = []
    all_ignore_map = []
    for lvl_id in range(num_lvls):
        h, w = featmap_sizes[lvl_id]
        loc_targets = torch.zeros(
            img_per_gpu,
            1,
            h,
            w,
            device=gt_bboxes_list[0].device,
            dtype=torch.float32)
        loc_weights = torch.full_like(loc_targets, -1)
        ignore_map = torch.zeros_like(loc_targets)
        all_loc_targets.append(loc_targets)
        all_loc_weights.append(loc_weights)
        all_ignore_map.append(ignore_map)
    for img_id in range(img_per_gpu):
        gt_bboxes = gt_bboxes_list[img_id]
        scale = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) *
                           (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1))
        min_anchor_size = scale.new_full(
            (1, ), float(anchor_scale * anchor_strides[0]))
        # assign gt bboxes to different feature levels w.r.t. their scales
        target_lvls = torch.floor(
            torch.log2(scale) - torch.log2(min_anchor_size) + 0.5)
        target_lvls = target_lvls.clamp(min=0, max=num_lvls - 1).long()
        for gt_id in range(gt_bboxes.size(0)):
            lvl = target_lvls[gt_id].item()
            # rescaled to corresponding feature map
            gt_ = gt_bboxes[gt_id, :4] / anchor_strides[lvl]
            # calculate ignore regions
            ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(
                gt_, r2, featmap_sizes[lvl])
            # calculate positive (center) regions
            ctr_x1, ctr_y1, ctr_x2, ctr_y2 = calc_region(
                gt_, r1, featmap_sizes[lvl])
            all_loc_targets[lvl][img_id, 0, ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 +
                                 1] = 1
            all_loc_weights[lvl][img_id, 0, ignore_y1:ignore_y2 +
                                 1, ignore_x1:ignore_x2 + 1] = 0
            all_loc_weights[lvl][img_id, 0, ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 +
                                 1] = 1
            # calculate ignore map on nearby low level feature
            if lvl > 0:
                d_lvl = lvl - 1
                # rescaled to corresponding feature map
                gt_ = gt_bboxes[gt_id, :4] / anchor_strides[d_lvl]
                ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(
                    gt_, r2, featmap_sizes[d_lvl])
                all_ignore_map[d_lvl][img_id, 0, ignore_y1:ignore_y2 +
                                      1, ignore_x1:ignore_x2 + 1] = 1
            # calculate ignore map on nearby high level feature
            if lvl < num_lvls - 1:
                u_lvl = lvl + 1
                # rescaled to corresponding feature map
                gt_ = gt_bboxes[gt_id, :4] / anchor_strides[u_lvl]
                ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(
                    gt_, r2, featmap_sizes[u_lvl])
                all_ignore_map[u_lvl][img_id, 0, ignore_y1:ignore_y2 +
                                      1, ignore_x1:ignore_x2 + 1] = 1
    for lvl_id in range(num_lvls):
        # ignore negative regions w.r.t. ignore map
        all_loc_weights[lvl_id][(all_loc_weights[lvl_id] < 0)
                                & (all_ignore_map[lvl_id] > 0)] = 0
        # set negative regions with weight 0.1
        all_loc_weights[lvl_id][all_loc_weights[lvl_id] < 0] = 0.1
    # loc average factor to balance loss
    loc_avg_factor = sum(
        [t.size(0) * t.size(-1) * t.size(-2) for t in all_loc_targets]) / 200
    return all_loc_targets, all_loc_weights, loc_avg_factor


def ga_shape_target(approx_list,
                    inside_flag_list,
                    square_list,
                    gt_bboxes_list,
                    img_metas,
                    approxs_per_octave,
                    cfg,
                    gt_bboxes_ignore_list=None,
                    sampling=True,
                    unmap_outputs=True):
    """Compute guided anchoring targets.

    Args:
        approx_list (list[list]): Multi level approxs of each image.
        inside_flag_list (list[list]): Multi level inside flags of each image.
        square_list (list[list]): Multi level squares of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        approxs_per_octave (int): number of approxs per octave
        cfg (dict): RPN train configs.
        gt_bboxes_ignore_list (list[Tensor]): ignore list of gt bboxes.
        sampling (bool): sampling or not.
        unmap_outputs (bool): unmap outputs or not.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(approx_list) == len(inside_flag_list) == len(
        square_list) == num_imgs
    # anchor number of multi levels
    num_level_squares = [squares.size(0) for squares in square_list[0]]
    # concat all level anchors and flags to a single tensor
    inside_flag_flat_list = []
    approx_flat_list = []
    square_flat_list = []
    for i in range(num_imgs):
        assert len(square_list[i]) == len(inside_flag_list[i])
        inside_flag_flat_list.append(torch.cat(inside_flag_list[i]))
        approx_flat_list.append(torch.cat(approx_list[i]))
        square_flat_list.append(torch.cat(square_list[i]))

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    (all_bbox_anchors, all_bbox_gts, all_bbox_weights, pos_inds_list,
     neg_inds_list) = multi_apply(
         ga_shape_target_single,
         approx_flat_list,
         inside_flag_flat_list,
         square_flat_list,
         gt_bboxes_list,
         gt_bboxes_ignore_list,
         img_metas,
         approxs_per_octave=approxs_per_octave,
         cfg=cfg,
         sampling=sampling,
         unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([bbox_anchors is None for bbox_anchors in all_bbox_anchors]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    bbox_anchors_list = images_to_levels(all_bbox_anchors, num_level_squares)
    bbox_gts_list = images_to_levels(all_bbox_gts, num_level_squares)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_squares)
    return (bbox_anchors_list, bbox_gts_list, bbox_weights_list, num_total_pos,
            num_total_neg)


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


def ga_shape_target_single(flat_approxs,
                           inside_flags,
                           flat_squares,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           img_meta,
                           approxs_per_octave,
                           cfg,
                           sampling=True,
                           unmap_outputs=True):
    """Compute guided anchoring targets.

    This function returns sampled anchors and gt bboxes directly
    rather than calculates regression targets.

    Args:
        flat_approxs (Tensor): flat approxs of a single image,
            shape (n, 4)
        inside_flags (Tensor): inside flags of a single image,
            shape (n, ).
        flat_squares (Tensor): flat squares of a single image,
            shape (approxs_per_octave * n, 4)
        gt_bboxes (Tensor): Ground truth bboxes of a single image.
        img_meta (dict): Meta info of a single image.
        approxs_per_octave (int): number of approxs per octave
        cfg (dict): RPN train configs.
        sampling (bool): sampling or not.
        unmap_outputs (bool): unmap outputs or not.

    Returns:
        tuple
    """
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    expand_inside_flags = inside_flags[:, None].expand(
        -1, approxs_per_octave).reshape(-1)
    approxs = flat_approxs[expand_inside_flags, :]
    squares = flat_squares[inside_flags, :]

    bbox_assigner = build_assigner(cfg.ga_assigner)
    assign_result = bbox_assigner.assign(approxs, squares, approxs_per_octave,
                                         gt_bboxes, gt_bboxes_ignore)
    if sampling:
        bbox_sampler = build_sampler(cfg.ga_sampler)
    else:
        bbox_sampler = PseudoSampler()
    sampling_result = bbox_sampler.sample(assign_result, squares, gt_bboxes)

    bbox_anchors = torch.zeros_like(squares)
    bbox_gts = torch.zeros_like(squares)
    bbox_weights = torch.zeros_like(squares)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        bbox_anchors[pos_inds, :] = sampling_result.pos_bboxes
        bbox_gts[pos_inds, :] = sampling_result.pos_gt_bboxes
        bbox_weights[pos_inds, :] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_squares.size(0)
        bbox_anchors = unmap(bbox_anchors, num_total_anchors, inside_flags)
        bbox_gts = unmap(bbox_gts, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (bbox_anchors, bbox_gts, bbox_weights, pos_inds, neg_inds)
