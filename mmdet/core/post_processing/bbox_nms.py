import torch
import torch.onnx.symbolic_helper as sym_help
from torch.autograd import Function
from torch.onnx.symbolic_opset9 import reshape
from torch.onnx.symbolic_opset10 import _slice

from mmdet.core.utils.misc import topk
from mmdet.ops.nms import nms_wrapper


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (float): NMS operation config
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
    combined_bboxes = GenericMulticlassNMS.apply(multi_bboxes, scores,
                                                 score_thr, nms_cfg, max_num)
    _, topk_inds = topk(combined_bboxes[:, 4].view(-1), max_num)
    combined_bboxes = combined_bboxes[topk_inds]
    bboxes = combined_bboxes[:, :5]
    labels = combined_bboxes[:, 5].long().view(-1)
    return bboxes, labels


class GenericMulticlassNMS(Function):

    @staticmethod
    def forward(ctx,
                multi_bboxes,
                multi_scores,
                score_thr,
                nms_cfg,
                max_num=-1):
        nms_op_cfg = nms_cfg.copy()
        nms_op_type = nms_op_cfg.pop('type', 'nms')
        nms_op = getattr(nms_wrapper, nms_op_type)

        num_classes = multi_scores.shape[1]
        bboxes, labels = [], []
        for i in range(num_classes):
            cls_inds = multi_scores[:, i] > score_thr
            if not cls_inds.any():
                continue
            # Get bboxes and scores of this class.
            if multi_bboxes.shape[1] == 4:
                _bboxes = multi_bboxes[cls_inds, :]
            else:
                _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
            _scores = multi_scores[cls_inds, i]
            cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
            cls_dets, _ = nms_op(cls_dets, **nms_op_cfg)
            cls_labels = multi_bboxes.new_full((cls_dets.shape[0],),
                                               i,
                                               dtype=torch.long)
            bboxes.append(cls_dets)
            labels.append(cls_labels)

        if bboxes:
            bboxes = torch.cat(bboxes)
            labels = torch.cat(labels)
            if bboxes.shape[0] > max_num:
                _, inds = bboxes[:, -1].topk(max_num, sorted=True)
                bboxes = bboxes[inds]
                labels = labels[inds]
            combined_bboxes = torch.cat(
                [bboxes, labels.to(bboxes.dtype).unsqueeze(-1)], dim=1)
        else:
            combined_bboxes = torch.zeros((0, 6), dtype=multi_bboxes.dtype, device=multi_bboxes.device)

        return combined_bboxes

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
