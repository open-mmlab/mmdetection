# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops.nms import batched_nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    """用于多类别 boxes 的 NMS.
        一般的reg值与cls值是解耦的,即对于网络输出层来说一个anchor/point输出4个reg值
        同时输出nc个cls值,此时reg与cls是无关的.
        但当这二者耦合在一起时,对网络输出层来说一个anchor/point就输出nc*4个reg值,
        同时输出nc个cls值,此时reg与cls是一一对应的.

    Args:
        multi_bboxes (Tensor): shape [n, nc*4] or [n, 4]
        multi_scores (Tensor): shape [n, nc],最后一列为背景score,但这将被忽略.
        score_thr (float): box 阈值,分数低于该阈值的 box 将不被考虑.
        nms_cfg (dict): nms的配置表
        max_num (int, optional): 截取nms后前max_num个box. -1代表不截取.
        score_factors (Tensor, optional): NMS之前,score将与其相乘.
        return_inds (bool, optional): 是否返回保留的box的索引.

    Returns:
        tuple: (dets, labels, indices (optional)),各个shape为 (k, 5), (k), (k).
        5 -> (x1, y1, x2, y2, score). labels∈[0,num_class).
    """
    num_classes = multi_scores.size(1) - 1
    # 忽略背景类别
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:  # 如果仅仅是单一类的box.为了兼容性,需要先增维再复制
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # 在TensorRT中并不支持NonZero操作, 过滤score低于score_thr的box
        valid_mask = scores > score_thr
    # 先过滤再乘以 score_factor 以保留更多 box, 在YOLOv3上 mAP 提高 1%
    if score_factors is not None:
        # 扩展至与score相同的shape,以方便与其相乘
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # .squeeze(1)的目的是nonzero在处理一维数据时会返回[m,1]格式的数据
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # 在TensorRT中并不支持NonZero操作,
        # TensorRT NMS 插件的无效输出会填充-1, 因此添加虚拟数据以使检测输出正确.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] 无法记录NMS, 因为box为空导致没有被执行')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], inds[keep]
    else:
        return dets, labels[keep]


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS 允许已经删除的box去抑制其他box,这样每个实例都可以并行决定保留或丢弃,这在传统的NMS
    中是不可能的,这种宽松的策略使我们能够完全在标准 GPU 加速矩阵运算中实现快速的NMS.值得注意的是
    每个box是可以预测nc个物体的,只要它们所属nc个不同类别.这也是有所区别于传统NMS的.

    Args:
        multi_bboxes (Tensor): shape [num_level*nms_pre, 4]
        multi_scores (Tensor): shape [num_level*nms_pre, nc+1],
            最后一列是背景类别的分数,但这将被忽略.
        multi_coeffs (Tensor): shape (num_level*nms_pre, num_protos).
        score_thr (float): box score阈值, box score低于该值得 box将被过滤.
        iou_thr (float): 过滤box时的iou阈值.
        top_k (int):如果在执行NMS之前有超过 top_k 个 box,那么只保留score最高的top_k个.
        max_num (int): 如果执行NMS之后有超过 max_num 个box,
            那么只保留score最高的max_num个, -1表示不进行该步操作.

    Returns:
        tuple: (dets, labels, coefficients)
            [max_per_img, 5], 具体格式为[x1, y1, x2, y2, cls_score]
            [max_per_img,], [max_per_img, num_protos].
    """

    scores = multi_scores[:, :-1].t()  # [nc, num_level*nms_pre] 剔除背景类
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [nc, top_k]
    num_classes, num_dets = idx.size()
    # 这里将idx [nc, top_k]表示在不同类情况下score最大的top_k个box索引,
    # 展平后作为索引,是为了将box在每个类上复制一份,因为box本身是与类无关的.
    # 同时也和scores在前两个维度对齐
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [nc, top_k, top_k]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)  # 对纵向的iou值进行坍塌取最大值 [nc, top_k]

    # 只过滤掉那些高于阈值的iou
    keep = iou_max <= iou_thr

    # 第二个阈值以可忽略的时间成本提升了 0.2 mAP
    keep *= scores > score_thr

    # 将每个筛选下来的box分配给其对应的类
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # 只保留所有类别中前max_num个最大score
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs
