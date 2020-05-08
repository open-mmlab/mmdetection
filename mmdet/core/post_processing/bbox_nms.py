import torch
from torch.autograd import Function

from mmdet.ops.nms import batched_nms


class MultiClassNMSFunction(Function):

    @staticmethod
    def forward(self,
                multi_bboxes,
                multi_scores,
                score_thr,
                iou_thr,
                max_num=-1,
                score_factors=None):
        if max_num > 0:
            multi_bboxes = multi_bboxes[:max_num]
            multi_scores = multi_scores[:max_num]
        return multi_bboxes, multi_scores

    @staticmethod
    def symbolic(g,
                 multi_bboxes,
                 multi_scores,
                 score_thr,
                 iou_thr,
                 max_num,
                 score_factors=None):
        num_detections, nmsed_boxes, nmsed_scores, nmsed_classes = g.op(
            'MultiClassNMS',
            multi_bboxes,
            multi_scores,
            score_threshold_f=score_thr,
            iou_threshold_f=iou_thr,
            top_k_i=2621,
            keepTopk_i=max_num,
            outputs=4)
        return nmsed_boxes, nmsed_classes


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
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    if max_num == -1:
        max_num = multi_bboxes.shape[0]
    # expand class dim to fit the size of TensorRT
    multi_bboxes = multi_bboxes.unsqueeze(1)
    multi_bboxes = multi_bboxes.expand(-1, multi_scores.shape[1], -1)
    # add batch dim
    multi_bboxes = multi_bboxes.unsqueeze(0)
    multi_scores = multi_scores.unsqueeze(0)
    return MultiClassNMSFunction.apply(multi_bboxes, multi_scores, score_thr,
                                       nms_cfg['iou_thr'], max_num)
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]
