import sys

import torch
import torch.onnx.symbolic_helper as sym_help
from torch.autograd import Function
from torch.onnx import is_in_onnx_export
from torch.onnx.symbolic_opset9 import reshape
from torch.onnx.symbolic_opset10 import _slice

from ..utils.misc import topk, dummy_pad
from ...ops.nms import batched_nms
from ...utils.deployment.symbolic import py_symbolic
from mmdet.integration.nncf import no_nncf_trace


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (dict): NMS operation config
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """

    if score_factors is not None:
        target_shape = list(score_factors.shape) + [1, ] * (multi_scores.dim() - score_factors.dim())
        scores = multi_scores * score_factors.view(*target_shape).expand_as(multi_scores)
    else:
        scores = multi_scores

    dets = multiclass_nms_core(multi_bboxes, scores, score_thr, nms_cfg, max_num)
    
    labels = dets[:, 5].long().view(-1)
    dets = dets[:, :5]

    return dets, labels


@py_symbolic()
def multiclass_nms_core(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1):
    num_classes = multi_scores.size(1)
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
    scores = multi_scores
        
    if is_in_onnx_export():
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
        return dets

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    labels = labels[keep]
    dets = torch.cat([dets, labels.to(dets.dtype).unsqueeze(-1)], dim=1)

    if not is_in_onnx_export() and max_num > 0:
        dets = dets[:max_num]

    return dets

