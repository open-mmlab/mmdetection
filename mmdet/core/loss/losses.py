# TODO merge naive and weighted loss.
import torch
import torch.nn.functional as F

from ..bbox import delta2bbox
from ...ops import sigmoid_focal_loss


def weighted_nll_loss(pred, label, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.nll_loss(pred, label, reduction='none')
    return torch.sum(raw * weight)[None] / avg_factor


def weighted_cross_entropy(pred, label, weight, avg_factor=None, reduce=True):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.cross_entropy(pred, label, reduction='none')
    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor


def weighted_binary_cross_entropy(pred, label, weight, avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    return F.binary_cross_entropy_with_logits(
        pred, label.float(), weight.float(),
        reduction='sum')[None] / avg_factor


def py_sigmoid_focal_loss(pred,
                          target,
                          weight,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                gamma=2.0,
                                alpha=0.25,
                                avg_factor=None,
                                num_classes=80):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return torch.sum(
        sigmoid_focal_loss(pred, target, gamma, alpha, 'none') * weight.view(
            -1, 1))[None] / avg_factor


def mask_cross_entropy(pred, target, label):
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='mean')[None]


def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor


def bounded_iou_loss(pred,
                     target_means,
                     target_stds,
                     rois,
                     gts,
                     weights,
                     beta=0.2,
                     avg_factor=None,
                     eps=1e-3):
    """Improving Object Localization with Fitness NMS and Bounded IoU Loss,
    https://arxiv.org/abs/1711.00164
    """

    inds = torch.nonzero(weights[:, 0] > 0)
    if avg_factor is None:
        avg_factor = inds.numel() + 1e-6

    if inds.numel() > 0:
        inds = inds.squeeze(1)
    else:
        return (pred * weights).sum()[None] / avg_factor

    pred_ = pred[inds, :]
    rois_ = rois[inds, :]
    gts_ = gts[inds, :]
    pred_bboxes = delta2bbox(
        rois_, pred_, target_means, target_stds, wh_ratio_clip=1e-6)
    pred_ctrx = (pred_bboxes[:, 0] + pred_bboxes[:, 2]) * 0.5
    pred_ctry = (pred_bboxes[:, 1] + pred_bboxes[:, 3]) * 0.5
    pred_w = pred_bboxes[:, 2] - pred_bboxes[:, 0] + 1
    pred_h = pred_bboxes[:, 3] - pred_bboxes[:, 1] + 1
    with torch.no_grad():
        gt_ctrx = (gts_[:, 0] + gts_[:, 2]) * 0.5
        gt_ctry = (gts_[:, 1] + gts_[:, 3]) * 0.5
        gt_w = gts_[:, 2] - gts_[:, 0] + 1
        gt_h = gts_[:, 3] - gts_[:, 1] + 1

    dx = gt_ctrx - pred_ctrx
    dy = gt_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (gt_w - 2 * dx.abs()) / (gt_w + 2 * dx.abs() + eps),
        torch.zeros_like(dx))
    loss_dy = 1 - torch.max(
        (gt_h - 2 * dy.abs()) / (gt_h + 2 * dy.abs() + eps),
        torch.zeros_like(dy))
    loss_dw = 1 - torch.min(gt_w / (pred_w + eps), pred_w / (gt_w + eps))
    loss_dh = 1 - torch.min(gt_h / (pred_h + eps), pred_h / (gt_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh], dim=-1).view(
        loss_dx.size(0), -1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)
    loss = loss.sum()[None] / avg_factor
    return loss


def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights
