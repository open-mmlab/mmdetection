import os

import torch


def dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape):
    """Clip boxes dynamically for onnx.

    Since torch.clamp cannot have dynamic
    `min` and `max`, we have to use torch.where to workaround.

    Args:
        x1 (Tensor): The x1 for bounding boxes.
        y1 (Tensor): The y1 for bounding boxes.
        x2 (Tensor): The x2 for bounding boxes.
        y2 (Tensor): The y2 for bounding boxes.
        max_shape (Tensor or torch.Size): The (H,W) of original image.
    Returns:
        tuple(Tensor): The clipped x1, y1, x2, y2.
    """
    assert isinstance(
        max_shape,
        torch.Tensor), '`max_shape` should be tensor of (h,w) for onnx'
    h = max_shape[0].to(x1)
    w = max_shape[1].to(x1)
    zero = x1.new_tensor(0)
    # clip by 0
    x1 = torch.where(x1 < zero, zero, x1)
    y1 = torch.where(y1 < zero, zero, y1)
    x2 = torch.where(x2 < zero, zero, x2)
    y2 = torch.where(y2 < zero, zero, y2)
    # clip by h and w
    x1 = torch.where(x1 > w, w, x1)
    y1 = torch.where(y1 > h, h, y1)
    x2 = torch.where(x2 > w, w, x2)
    y2 = torch.where(y2 > h, h, y2)
    return x1, y1, x2, y2


def get_k_for_topk(k, size):
    """Get k of TopK for onnx exporting.

    The K of TopK in TensorRT should not be a Tensor, while in ONNX Runtime
      it could be a Tensor.Due to dynamic shape feature, we have to decide
      whether to do TopK and what K it should be while exporting to ONNX.
    If returned K is less than zero, it means we do not have to do
      TopK operation.

    Args:
        k (int or Tensor): The set k value for nms from config file.
        size (Tensor or torch.Size): The number of elements of \
            TopK's input tensor
    Returns:
        tuple: (int or Tensor): The final K for TopK.
    """
    ret_k = -1
    if k <= 0 or size <= 0:
        return ret_k
    if torch.onnx.is_in_onnx_export():
        is_trt_backend = os.environ.get('ONNX_BACKEND') == 'MMCVTensorRT'
        if is_trt_backend:
            # TensorRT does not support dynamic K with TopK op
            if 0 < k < size:
                ret_k = k
        else:
            # Always keep topk op for dynamic input in onnx for ONNX Runtime
            ret_k = torch.where(k < size, k, size)
    elif k < size:
        ret_k = k
    else:
        # ret_k is -1
        pass
    return ret_k


def add_dummy_nms_for_onnx(boxes,
                           scores,
                           max_output_boxes_per_class=1000,
                           iou_threshold=0.5,
                           score_threshold=0.05,
                           pre_top_k=-1,
                           after_top_k=-1,
                           only_return_indices=False):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    This function helps exporting to onnx with batch and multiclass NMS op

    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4]
        scores (Tensor): The detection scores of shape \
            [N, num_boxes, num_classes]
        max_output_boxes_per_class (int): Maximum number of output \
            boxes per class of nms. Defaults to 1000
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5
        score_threshold (float): score threshold of nms. \
            Defaults to 0.05
        pre_top_k (bool): Number of top K boxes to keep before nms \
            Defaults to -1
        after_top_k (int): Number of top K boxes to keep after nms \
            Defaults to -1
        only_return_indices (bool): whether to only return selected \
            indices from nms. Defaults to False.

    Returns:
        tuple: (indices) or (dets, batch_inds, cls_inds) :
            If only_return_indices is True, this function returns
            the output of nms with shape of [N, 3], and each row's
            format is [batch_index, class_index, box_index].
            Otherwise, it would return dets of shape[N, 5], batch \
            indices of shape [N,] and class labels of shape [N,].
    """
    max_output_boxes_per_class = torch.LongTensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)
    batch_size = scores.shape[0]
    num_class = scores.shape[2]

    nms_pre = torch.tensor(pre_top_k, device=scores.device, dtype=torch.long)
    nms_pre = get_k_for_topk(nms_pre, boxes.shape[1])

    if nms_pre > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.topk(nms_pre)
        batch_inds = torch.arange(batch_size).view(
            -1, 1).expand_as(topk_inds).long()
        # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
        transformed_inds = boxes.shape[1] * batch_inds + topk_inds
        boxes = boxes.reshape(-1, 4)[transformed_inds, :].reshape(
            batch_size, -1, 4)
        scores = scores.reshape(-1, num_class)[transformed_inds, :].reshape(
            batch_size, -1, num_class)

    scores = scores.permute(0, 2, 1)
    num_box = boxes.shape[1]
    # turn off tracing to create a dummy output of nms
    state = torch._C._get_tracing_state()
    # dummy indices of nms's output
    batch_inds = torch.randint(batch_size, (num_box, 1))
    cls_inds = torch.randint(num_class, (num_box, 1))
    box_inds = torch.randint(num_box, (num_box, 1))
    indices = torch.cat([batch_inds, cls_inds, box_inds], dim=1)
    output = indices
    setattr(DymmyONNXNMSop, 'output', output)

    # open tracing
    torch._C._set_tracing_state(state)
    selected_indices = DymmyONNXNMSop.apply(boxes, scores,
                                            max_output_boxes_per_class,
                                            iou_threshold, score_threshold)
    if only_return_indices:
        return selected_indices

    batch_inds, cls_inds = selected_indices[:, 0], selected_indices[:, 1]
    box_inds = selected_indices[:, 2]
    labels = torch.arange(num_class, dtype=torch.long).to(scores.device)
    labels = labels.view(1, num_class, 1).expand_as(scores).reshape(-1, 1)
    scores = scores.reshape(-1, 1)
    boxes = boxes.reshape(batch_size, -1).repeat(1, num_class).reshape(-1, 4)
    pos_inds = (num_class * batch_inds + cls_inds) * num_box + box_inds
    mask = scores.new_zeros(scores.shape)
    # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
    # PyTorch style code: mask[batch_inds, box_inds] += 1
    mask[pos_inds, :] += 1
    scores = scores * mask
    boxes = boxes * mask

    scores = scores.reshape(batch_size, -1)
    boxes = boxes.reshape(batch_size, -1, 4)
    labels = labels.reshape(batch_size, -1)

    nms_after = torch.tensor(
        after_top_k, device=scores.device, dtype=torch.long)
    nms_after = get_k_for_topk(nms_after, num_box * num_class)

    if nms_after > 0:
        _, topk_inds = scores.topk(nms_after)
        batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds)
        # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
        transformed_inds = scores.shape[1] * batch_inds + topk_inds
        scores = scores.reshape(-1, 1)[transformed_inds, :].reshape(
            batch_size, -1)
        boxes = boxes.reshape(-1, 4)[transformed_inds, :].reshape(
            batch_size, -1, 4)
        labels = labels.reshape(-1, 1)[transformed_inds, :].reshape(
            batch_size, -1)

    scores = scores.unsqueeze(2)
    dets = torch.cat([boxes, scores], dim=2)
    return dets, labels


class DymmyONNXNMSop(torch.autograd.Function):
    """DymmyONNXNMSop.

    This class is only for creating onnx::NonMaxSuppression.
    """

    @staticmethod
    def forward(ctx, boxes, scores, max_output_boxes_per_class, iou_threshold,
                score_threshold):

        return DymmyONNXNMSop.output

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold,
                 score_threshold):
        return g.op(
            'NonMaxSuppression',
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            outputs=1)
