import torch


def add_dummy_nms_for_onnx(boxes,
                           scores,
                           max_output_boxes_per_class=1000,
                           iou_threshold=0.5,
                           score_threshold=0.05,
                           only_return_indices=False):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    This function helps exporting to onnx with batch and multiclass NMS op

    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4]
        scores (Tensor): The detection scores of shape \
            [N, num_classes, num_boxes]
        max_output_boxes_per_class (int): Maximum number of output \
            boxes per class of nms. Defaults to 1000
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5
        score_threshold (float): score threshold of nms. \
            Defaults to 0.05
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
    # turn off tracing
    state = torch._C._get_tracing_state()
    batch_size = scores.shape[0]
    num_class = scores.shape[1]
    num_box = scores.shape[2]
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
    boxes = boxes[batch_inds, box_inds, :]
    scores = scores[batch_inds, cls_inds, box_inds].unsqueeze(1)
    dets = torch.cat([boxes, scores], dim=1)
    return dets, batch_inds, cls_inds


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
