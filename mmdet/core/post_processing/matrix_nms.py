# Copyright (c) OpenMMLab. All rights reserved.
import torch


def mask_matrix_nms(masks,
                    labels,
                    scores,
                    filter_thr=-1,
                    nms_pre=-1,
                    max_num=-1,
                    kernel='gaussian',
                    sigma=2.0,
                    mask_area=None):
    """用于多类别的Matrix NMS.

    Args:
        masks (Tensor): [num_instances, h, w], bool值代表该位置的像素是否是实例.
        labels (Tensor): [num_instances, ]
        scores (Tensor): [num_instances, ]
        filter_thr (float): 在matrix nms 之后过滤mask的分数阈值。默认:-1,表示不过滤.
        nms_pre (int): 进行matrix nms之前的最大实例数量.默认:-1,表示不限制.
        max_num (int, optional): 进行matrix nms之后保存到的最大实例数量. 默认:-1,表示不限制.
        kernel (str): 'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
        mask_area (Tensor): [num_instances, ].表示任意实例在[h, w]上的像素数量

    Returns:
        tuple(Tensor): Processed mask results.

            - scores (Tensor): Updated scores, has shape (n,).
            - labels (Tensor): Remained labels, has shape (n,).
            - masks (Tensor): Remained masks, has shape (n, w, h).
            - keep_inds (Tensor): The indices number of
                the remaining mask in the input mask, has shape (n,).
    """
    assert len(labels) == len(masks) == len(scores)
    if len(labels) == 0:
        return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
            0, *masks.shape[-2:]), labels.new_zeros(0)
    if mask_area is None:
        mask_area = masks.sum((1, 2)).float()
    else:
        assert len(masks) == len(mask_area)

    # sort and keep top nms_pre
    scores, sort_inds = torch.sort(scores, descending=True)

    keep_inds = sort_inds
    if 0 < nms_pre < len(sort_inds):
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)  # [num_instances, h, w] -> [num_instances, h * w]
    flatten_masks = masks.reshape(num_masks, -1).float()
    # [num_instances, num_instances]
    # 要搞懂下面这行矩阵乘法的代码需要先明白flatten_mask代表的物理意义,以下以S代表num_grid
    # flatten_masks:[S**2, h*w],令i∈[0,S**2),j∈[0, h*w).
    # i表示网络在哪个空间位置进行预测,网络将输出分割成了S**2个位置,负责预测不同的分割结果
    # 并且每个位置仅能预测一个实例,而分割结果上的bool值也即代表该位置j是否属于空间位置i所对应的类别.
    # 因为在SOLO论文中提及,能不能通过实例(中心)位置或者实例大小来区分不同的实例,那么如何做呢?
    # 1.分辨出不同位置的实例,通过CoordConv来实现
    # 2.某一层级在生成正样本区域时,忽略那些(gt box的面积平方根)超过一定范围内(不同层级不同范围)
    # SOLO输出两个值,
    # pred_mask:[bs,S**2,h,w]. 表示网络在某个位置预测的分割结果
    # pred_score[bs, nc, S, S]. 表示网络在某个位置预测的cls_score
    # 比如inter_matrix矩阵右上角的值代表了flatten_masks第一维度上第一个与最后一个位置上的分割交集
    inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    # [num_instances,] -> [1, num_instances] -> [num_instances, num_instances]
    expanded_mask_area = mask_area.expand(num_masks, num_masks)
    # 上三角 iou 矩阵.
    iou_matrix = (inter_matrix /
                  (expanded_mask_area + expanded_mask_area.transpose(1, 0) -
                   inter_matrix)).triu(diagonal=1)
    # 将label也进行广播.
    expanded_labels = labels.expand(num_masks, num_masks)
    # 上三角 label 矩阵. 因为nms操作默认在同一类别之间进行,所以这里同类别之间的值为1
    label_matrix = (expanded_labels == expanded_labels.transpose(
        1, 0)).triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(num_masks,
                                           num_masks).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # Calculate the decay_coefficient
    # 这里最后要执行min(0)操作有两个原因
    # 1.matrix_nms是在不同类别之间进行的,当类别不同时,decay_iou是为0的.此时decay_coefficient对应
    # 位置的值就大于0,那么如果不取最小值(最小值上面有提,是0)就会让后续的score变大,可能超过1
    # 2.假设有a,b,c三个物体,那么就尽可能的选择iou_ab最小,iou_bc最大情况下,惩罚物体c的score值
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(
            f'{kernel} kernel is not supported in matrix nms!')
    # update the score.
    scores = scores * decay_coefficient

    # 对score根据filter_thr进行过滤
    if filter_thr > 0:
        keep = scores >= filter_thr
        keep_inds = keep_inds[keep]
        if not keep.any():
            return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
                0, *masks.shape[-2:]), labels.new_zeros(0)
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

    # 因为上面对score进行了不同程度的惩罚,所以这里需要重新排序
    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    if 0 < max_num < len(sort_inds):
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    return scores, labels, masks, keep_inds
