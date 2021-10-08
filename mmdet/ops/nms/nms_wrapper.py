import numpy as np
import sys
import torch
from mmcv.utils import deprecated_api_warning

from mmdet.integration.nncf.utils import is_in_nncf_tracing, no_nncf_trace


class NMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx, bboxes, scores, iou_threshold, score_threshold, max_num,
                offset):
        with no_nncf_trace():
            valid_mask = scores > score_threshold
        bboxes, scores = bboxes[valid_mask], scores[valid_mask]
        from mmcv.ops.nms import NMSop
        inds = NMSop.forward(ctx, bboxes, scores, iou_threshold, offset,
                             score_threshold, max_num)
        inds = inds[:max_num]
        return inds

    @staticmethod
    def symbolic(g, bboxes, scores, iou_threshold, score_threshold, max_num,
                 offset):
        from mmdet.utils.deployment.symbolic import nms_symbolic_with_score_thr
        return nms_symbolic_with_score_thr(g, bboxes, scores, iou_threshold,
                                           score_threshold, max_num, offset)


@deprecated_api_warning({'iou_thr': 'iou_threshold'})
def nms(boxes, scores, iou_threshold, score_thr=0.0, max_num=-1, offset=0):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either torch tensor or numpy array. GPU NMS will be used
    if the input is gpu tensor, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        boxes (torch.Tensor or np.ndarray): boxes in shape (N, 4).
        scores (torch.Tensor or np.ndarray): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        score_thr (float): score threshold for NMS.
        max_num (int): maximum number of boxes after NMS.
        offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).

    Returns:
        tuple: kept dets(boxes and scores) and indice, which is always the \
            same data type as the input.

    Example:
        >>> boxes = np.array([[49.1, 32.4, 51.0, 35.9],
        >>>                   [49.3, 32.9, 51.0, 35.3],
        >>>                   [49.2, 31.8, 51.0, 35.4],
        >>>                   [35.1, 11.5, 39.1, 15.7],
        >>>                   [35.6, 11.8, 39.3, 14.2],
        >>>                   [35.3, 11.5, 39.9, 14.5],
        >>>                   [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
        >>> scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],\
               dtype=np.float32)
        >>> iou_threshold = 0.6
        >>> dets, inds = nms(boxes, scores, iou_threshold)
        >>> assert len(inds) == len(dets) == 3
    """
    assert isinstance(boxes, (torch.Tensor, np.ndarray))
    assert isinstance(scores, (torch.Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)

    if torch.__version__ == 'parrots':
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + offset) * (y2 - y1 + offset)
        _, order = scores.sort(0, descending=True)
        if boxes.device == 'cpu':
            indata_list = [boxes, order, areas]
            indata_dict = {
                'iou_threshold': float(iou_threshold),
                'offset': int(offset)
            }
            select = ext_module.nms(*indata_list, **indata_dict).byte()
        else:
            boxes_sorted = boxes.index_select(0, order)
            indata_list = [boxes_sorted, order, areas]
            indata_dict = {
                'iou_threshold': float(iou_threshold),
                'offset': int(offset)
            }
            select = ext_module.nms(*indata_list, **indata_dict)
        inds = order.masked_select(select)
    else:
        if max_num < 0:
            max_num = boxes.shape[0]
        inds = NMSop.apply(boxes, scores, iou_threshold, score_thr, max_num,
                           offset)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.

            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.

    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        with no_nncf_trace():
            # NB: this trick is required to make class-separate NMS using ONNX NMS operation;
            #     the ONNX NMS operation supports another way of class separation (class-separate scores), but this is not used here.
            # Note that `not_nncf_trace` is required here, since this trick causes accuracy degradation in case of int8 quantization:
            #      if the output of the addition below is quantized, the maximal output value is about
            #      ~ max_value_in_inds * max_coordinate,
            #      usually this value is big, so after int8-quantization different small bounding
            #      boxes may be squashed into the same bounding box, this may cause accuracy degradation.
            # TODO: check if it is possible in this architecture use class-separate scores that are supported in ONNX NMS.
            boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr or (torch.onnx.is_in_onnx_export()
                                              or is_in_nncf_tracing):
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        scores = dets[:, 4]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]
    return torch.cat([boxes, scores[:, None]], -1), keep
