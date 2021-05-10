import torch
from mmcv.ops.nms import batched_nms, nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.post_processor.builder import POST_PROCESSOR


@POST_PROCESSOR.register_module()
class PreNMS(object):
    """Reshape the prediction and remove low score boxes before the NMS.

    Args:
        score_thr (float): bbox threshold, boxes with scores
            lower than would be removed. Defaults to 0.05.
        score_factor_thr (float, optional): score_factor threshold,
            boxes with score_factor lower than would be removed.
            Defaults to None.
    """

    def __init__(self, score_thr=0.05, score_factor_thr=None):
        self.score_thr = score_thr
        self.score_factor_thr = score_factor_thr

    def __call__(self, results_list):
        # shape(n, num_class), the num_class does not
        # include the background
        r_results_list = []
        for results in results_list:
            # exclude background category
            scores = results.scores[..., :-1]
            # multi_bboxes (Tensor): shape (n, num_class*4) or (n, 4)
            multi_bboxes = results.bboxes
            score_factors = results.get('score_factors', None)
            num_classes = scores.size(1)
            if multi_bboxes.shape[1] > 4:
                bboxes = multi_bboxes.view(scores.size(0), -1, 4)
            else:
                bboxes = multi_bboxes[:, None].expand(
                    scores.size(0), num_classes, 4)

            labels = torch.arange(num_classes, dtype=torch.long)
            labels = labels.view(1, -1).expand_as(scores)

            bboxes = bboxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            r_results = results.new_results()
            r_results.bboxes = bboxes
            r_results.scores = scores
            r_results.labels = labels
            r_results.keep_ids = torch.arange(
                0, len(r_results), device=scores.device)

            if not torch.onnx.is_in_onnx_export():
                # NonZero not supported  in TensorRT
                # remove low scoring boxes
                valid_mask = scores > self.score_thr

            # multiply score_factor after threshold to preserve
            # more bboxes, improve mAP by 1% for YOLOv3
            if score_factors is not None:
                # expand the shape to match original shape of score
                score_factors = score_factors.view(-1,
                                                   1).repeat(1, num_classes)
                score_factors = score_factors.reshape(-1)
                r_results.scores = r_results.scores * score_factors

                if self.score_factor_thr:
                    # YOLO  has a score_factor_thr
                    valid_mask = valid_mask & (
                        score_factors > self.score_factor_thr)

            if not torch.onnx.is_in_onnx_export():
                # NonZero not supported  in TensorRT
                r_results = r_results[valid_mask]
            else:
                # TensorRT NMS plugin has invalid output filled with -1
                # add dummy data to make detection output correct.
                assert bboxes.size(1) == 4, 'TensorRT NMS plugin only ' \
                                            'support class ignore' \
                                            'regression'
                bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
                scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
                labels = torch.cat([labels, labels.new_zeros(1)], dim=0)
                r_results = results.new_results()
                r_results.bboxes = bboxes
                r_results.labels = labels
                r_results.scores = scores

            if len(r_results) == 0:
                if torch.onnx.is_in_onnx_export():
                    raise RuntimeError('[ONNX Error] Can not record NMS '
                                       'as it has not been executed this time')
            r_results_list.append(r_results)
        return r_results_list


@POST_PROCESSOR.register_module()
class NaiveNMS(object):

    def __init__(self,
                 iou_threshold=0.5,
                 class_agnostic=False,
                 max_num=100,
                 split_thr=10000,
                 offset=0):
        self.iou_threshold = iou_threshold
        self.class_agnostic = class_agnostic
        self.max_num = max_num
        self.split_thr = split_thr
        self.offset = offset

    def __call__(self, results_list):
        r_results_list = []

        for results in results_list:
            if len(results) == 0:
                r_results_list.append(results)
                continue

            bboxes = results.bboxes
            labels = results.labels
            scores = results.scores

            if self.class_agnostic:
                boxes_for_nms = bboxes
            else:
                max_coordinate = bboxes.max()
                offsets = labels.to(bboxes) * (
                    max_coordinate + torch.tensor(1).to(bboxes))
                boxes_for_nms = bboxes + offsets[:, None]

            if boxes_for_nms.shape[0] < self.split_thr or \
                    torch.onnx.is_in_onnx_export():
                dets, keep_ids = nms(
                    boxes_for_nms,
                    scores,
                    iou_threshold=self.iou_threshold,
                    offset=self.offset)
                r_results = results[keep_ids]

            else:
                total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
                for id in torch.unique(labels):
                    class_ids = (labels == id).nonzero(as_tuple=False).view(-1)
                    dets, keep_ids = nms(
                        boxes_for_nms[class_ids],
                        scores[class_ids],
                        iou_threshold=self.iou_threshold,
                        offset=self.offset)
                    total_mask[class_ids[keep_ids]] = True

                keep = total_mask.nonzero(as_tuple=False).view(-1)
                keep = keep[scores[keep].argsort(descending=True)]
                r_results = results[keep]
            if self.max_num > 0:
                r_results = r_results[:self.max_num]
            r_results_list.append(r_results)

        return r_results_list


@POST_PROCESSOR.register_module()
class ScoreTopk(object):
    """Select bboxes with top `max_per_img` score.

    Args:
        sigmoid (bool): The score should be activated with sigmoid or softmax.
            Default to True.
        max_per_image (int): Max
    """

    def __init__(
        self,
        sigmoid=True,
        max_per_img=100,
    ):
        self.with_sigmoid = sigmoid
        self.max_per_img = max_per_img

    def __call__(self, results_list):

        r_results_list = []
        for results in results_list:
            if len(results) == 0:
                r_results_list.append(results)
                continue

            r_results = results.new_results()
            bboxes = results.bboxes
            scores = results.scores

            if self.with_sigmoid:
                # without background padding
                num_class = scores.size(-1)
                cls_scores = scores.sigmoid()
                cls_scores, indexs = cls_scores.view(-1).topk(self.max_per_img)
                det_labels = indexs % num_class
                bbox_index = indexs // num_class
                r_results.labels = det_labels
                r_results.scores = cls_scores
                r_results.bboxes = bboxes[bbox_index]

            else:
                cls_score, det_labels = scores.softmax(-1)[..., :-1].max(-1)
                cls_score, bbox_index = cls_score.topk(self.max_per_img)
                bbox_pred = bboxes[bbox_index]
                det_labels = det_labels[bbox_index]
                r_results.scores = cls_score
                r_results.bboxes = bbox_pred
                r_results.labels = det_labels

            r_results_list.append(r_results)

        return r_results_list


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
        nms_thr (float): NMS IoU threshold
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
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        if return_inds:
            return bboxes, labels, inds
        else:
            return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], keep
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
