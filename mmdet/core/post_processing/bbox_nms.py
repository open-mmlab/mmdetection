import torch
import torch.onnx.symbolic_helper as sym_help
from torch.autograd import Function
from torch.onnx import is_in_onnx_export
from torch.onnx.symbolic_opset9 import reshape
from torch.onnx.symbolic_opset10 import _slice

from ..utils.misc import topk
from ...ops.nms import batched_nms


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

    dets = MulticlassNMS.apply(multi_bboxes, scores, score_thr, nms_cfg, max_num)

    if max_num > 0:
        if is_in_onnx_export():
            scores = dets[:, 4].view(-1)
            _, topk_inds = topk(scores, max_num)
            dets = dets[topk_inds]
        else:
            dets = dets[:max_num]
    
    labels = dets[:, 5].long().view(-1)
    dets = dets[:, :5]

    return dets, labels


class MulticlassNMS(Function):

    @staticmethod
    def forward(ctx,
                multi_bboxes,
                multi_scores,
                score_thr,
                nms_cfg,
                max_num=-1):
        if is_in_onnx_export():
            assert nms_cfg['type'] == 'nms', 'Only vanilla NMS is compatible with ONNX export'

        if multi_bboxes.shape[1] > 4:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        else:
            num_classes = multi_scores.size(1)
            bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
            
        # filter out boxes with low scores
        valid_mask = multi_scores > score_thr
        bboxes = bboxes[valid_mask]
        scores = multi_scores[valid_mask]
        labels = valid_mask.nonzero()[:, 1]

        if bboxes.numel() == 0:
            dets = multi_bboxes.new_zeros((0, 6))
            return dets

        dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

        labels = labels[keep]
        dets = torch.cat([dets, labels.to(dets.dtype).unsqueeze(-1)], dim=1)
        return dets

    @staticmethod
    def symbolic(g,
                 multi_bboxes,
                 multi_scores,
                 score_thr,
                 nms_cfg,
                 max_num=-1):

        def cast(x, dtype):
            return g.op('Cast', x, to_i=sym_help.cast_pytorch_to_onnx[dtype])

        def get_size(x, dim):
            shape = g.op('Shape', x)
            dim = _slice(g, shape, axes=[0], starts=[dim], ends=[dim + 1])
            return cast(dim, 'Long')

        nms_op_type = nms_cfg.get('type', 'nms')
        assert nms_op_type == 'nms'
        assert 'iou_thr' in nms_cfg
        iou_threshold = nms_cfg['iou_thr']
        assert 0 <= iou_threshold <= 1

        # Transpose and reshape input tensors to fit ONNX NonMaxSuppression.
        multi_bboxes = reshape(g, multi_bboxes, [0, -1, 4])
        multi_bboxes = g.op('Transpose', multi_bboxes, perm_i=[1, 0, 2])

        batches_num = get_size(multi_bboxes, 0)
        spatial_num = get_size(multi_bboxes, 1)

        multi_scores = g.op('Transpose', multi_scores, perm_i=[1, 0])
        scores_shape = g.op('Concat',
                            batches_num,
                            g.op('Constant', value_t=torch.LongTensor([-1])),
                            spatial_num, axis_i=0)
        multi_scores = reshape(g, multi_scores, scores_shape)
        classes_num = get_size(multi_scores, 1)

        assert max_num > 0

        indices = g.op(
            'NonMaxSuppression', multi_bboxes, multi_scores,
            g.op('Constant', value_t=torch.LongTensor([max_num])),
            g.op('Constant', value_t=torch.FloatTensor([iou_threshold])),
            g.op('Constant', value_t=torch.FloatTensor([score_thr])))

        # Flatten bboxes and scores.
        multi_bboxes_flat = reshape(g, multi_bboxes, [-1, 4])
        multi_scores_flat = reshape(g, multi_scores, [-1, ])

        # Flatten indices.
        batch_indices = cast(_slice(g, indices, axes=[1], starts=[0], ends=[1]), 'Long')
        class_indices = cast(_slice(g, indices, axes=[1], starts=[1], ends=[2]), 'Long')
        box_indices = cast(_slice(g, indices, axes=[1], starts=[2], ends=[3]), 'Long')

        def add(*args, dtype='Long'):
            x = g.op('Add', args[0], args[1])
            if dtype is not None:
                x = cast(x, dtype)
            return x

        def mul(*args, dtype='Long'):
            x = g.op('Mul', args[0], args[1])
            if dtype is not None:
                x = cast(x, dtype)
            return x

        flat_box_indices = add(mul(batch_indices, spatial_num), box_indices)
        flat_score_indices = add(mul(add(mul(batch_indices, classes_num), class_indices), spatial_num), box_indices)

        # Select bboxes.
        out_bboxes = reshape(
            g, g.op('Gather', multi_bboxes_flat, flat_box_indices, axis_i=0),
            [-1, 4])
        out_scores = reshape(
            g, g.op('Gather', multi_scores_flat, flat_score_indices, axis_i=0),
            [-1, 1])
        # Having either batch size or number of classes here equal to one is the limitation of implementation.
        class_indices = reshape(g, cast(add(class_indices, batch_indices), 'Float'), [-1, 1])

        # Combine bboxes, scores and labels into a single tensor.
        # This a workaround for a PyTorch bug (feature?),
        # limiting ONNX operations to output only single tensor.
        out_combined_bboxes = g.op(
            'Concat', out_bboxes, out_scores, class_indices, axis_i=1)

        return out_combined_bboxes
