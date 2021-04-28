import torch
from torch.onnx import is_in_onnx_export

from mmdet.integration.nncf import no_nncf_trace, is_in_nncf_tracing

from mmdet.ops.nms import batched_nms

from ...utils.deployment.symbolic import py_symbolic
from ..utils.misc import dummy_pad, topk
from mmdet.core.bbox.iou_calculators import bbox_overlaps


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (dict): NMS operation config
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (bboxes, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Labels are 0-based.
    """

    if score_factors is not None:
        target_shape = list(score_factors.shape) + [1, ] * (multi_scores.dim() - score_factors.dim())
        scores = multi_scores * score_factors.view(*target_shape).expand_as(multi_scores)
    else:
        scores = multi_scores

    dets, keep = multiclass_nms_core(multi_bboxes, scores, score_thr, nms_cfg, max_num, return_inds)
    
    labels = dets[:, 5].long().view(-1)
    dets = dets[:, :5]

    if keep is not None:
        return dets, labels, keep
    else:
        return dets, labels


@py_symbolic()
def multiclass_nms_core(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, return_inds=False):
    num_classes = multi_scores.size(1)
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
    scores = multi_scores
        
    if is_in_onnx_export() or is_in_nncf_tracing():
        with no_nncf_trace():
            labels = torch.arange(num_classes, dtype=torch.long, device=scores.device) \
                          .unsqueeze(0) \
                          .expand_as(scores) \
                          .reshape(-1)
            bboxes = bboxes.reshape(-1, 4)
            scores = scores.reshape(-1)

        assert nms_cfg['type'] == 'nms', 'Only vanilla NMS is compatible with ONNX export'
        nms_cfg['score_thr'] = score_thr
        nms_cfg['max_num'] = max_num if max_num > 0 else sys.maxsize
    else:
        with no_nncf_trace():
            valid_mask = scores > score_thr
        bboxes = bboxes[valid_mask]
        scores = scores[valid_mask]
        labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        dets = multi_bboxes.new_zeros((0, 6))
        return dets, None

    # TODO: add size check before feed into batched_nms
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    labels = labels[keep]
    dets = torch.cat([dets, labels.to(dets.dtype).unsqueeze(-1)], dim=1)

    if not (is_in_onnx_export() or is_in_nncf_tracing()) and max_num > 0:
        dets = dets[:max_num]

    if return_inds:
        return dets, keep
    else:
        return dets, None


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (bboxes, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs
