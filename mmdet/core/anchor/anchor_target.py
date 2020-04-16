import torch

from ..bbox import PseudoSampler, assign_and_sample, bbox2delta, build_assigner
from ..utils import multi_apply


def anchor_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True):
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
    num_imgs = len(img_metas)#图片个数
    assert len(anchor_list) == len(valid_flag_list) == num_imgs #保证图片个数和标签个数全部一致

    # anchor number of multi levels特征层数
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor将所有层的anchor还有他的标志全部组合成一个tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:#有效图片
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         anchor_target_single,
         anchor_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs)
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
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.
        将图片标签GT转化到各个特征图上
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


def anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)#筛出图片中没有超出边界的anchors
    if not inside_flags.any():#如果没有满足的，就返回空的，
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]#取出在图片范围内的anchor

    if sampling:#如果要对anchor进行采样，用于二阶段
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:#如果不需要对anchors进行采样，用于一阶段
        bbox_assigner = build_assigner(cfg.assigner)#就直接进行分配
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)
    #设置好所有标签的大小，用于填充
    num_valid_anchors = anchors.shape[0]#有效anchors的数量
    bbox_targets = torch.zeros_like(anchors)#目标准备
    bbox_weights = torch.zeros_like(anchors)#每个ancors的权值
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds    #正样本索引
    neg_inds = sampling_result.neg_inds     #负样本索引
    if len(pos_inds) > 0:#如果有正样本
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)#对正样本和GT进行delta编码
        bbox_targets[pos_inds, :] = pos_bbox_targets #将其放入到正样本位置
        bbox_weights[pos_inds, :] = 1.0   #正样本权重设为1
        if gt_labels is None: #如果没有GT，那么
            labels[pos_inds] = 1  
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:#如果有负样本，权重设为1，这个不是标签
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:#这个是将
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


def anchor_inside_flags(flat_anchors,
                        valid_flags,
                        img_shape,
                        allowed_border=0):#对anchor进行规范化，超出边界的anchor索引要设为无效，
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 1] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 2] < img_w + allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 3] < img_h + allowed_border).type(torch.uint8)
    else:
        inside_flags = valid_flags #返回在图片中的有效anchors索引
    return inside_flags


def unmap(data, count, inds, fill=0):#将数据reshape,并将它放在指定的索引位置,用数据进行填充
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:#如果数据是一维的
        ret = data.new_full((count, ), fill)#设置好返回数据的大小
        ret[inds] = data#指定的inds也是data，将data放在相应的位置
    else:
        new_size = (count, ) + data.size()[1:]#多维数据放置也用相同的方法，类似于进行广播，但是只有一个位置有数据
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
