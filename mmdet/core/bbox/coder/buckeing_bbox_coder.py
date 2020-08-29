import numpy as np
import torch
import torch.nn.functional as F

from ..builder import BBOX_CODERS
from ..transforms import bbox_rescale
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class BucketingBBoxCoder(BaseBBoxCoder):
    """Bucketing BBox Coder for Side-Aware Bounday Localization (SABL).

    Boundary Localization with Bucketing and Bucketing Guided Rescoring
    are implemented here.

    Please refer to https://arxiv.org/abs/1912.04260 for more details.

    Args:
        num_buckets (int): Number of buckets.
        scale_factor (int): Scale factor of proposals to generate buckets.
        offset_topk (int): Topk buckets are used to generate \
            bucket fine regression targets. Defaults to 2.
        offset_allow (float): Offset allowance to generate \
            bucket fine regression targets. \
            To avoid too large offset displacements. Defaults to 1.0.
        cls_ignore_neighbor (bool): Ignore second nearest bucket or Not. \
            Defaults to True.
    """

    def __init__(self,
                 num_buckets,
                 scale_factor,
                 offset_topk=2,
                 offset_allow=1.0,
                 cls_ignore_neighbor=True):
        super(BucketingBBoxCoder, self).__init__()
        self.num_buckets = num_buckets
        self.scale_factor = scale_factor
        self.offset_topk = offset_topk
        self.offset_allow = offset_allow
        self.cls_ignore_neighbor = cls_ignore_neighbor

    def encode(self, bboxes, gt_bboxes):
        """Get bucketing estimation and fine regression targets during
        training."""

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = bbox2bucket(bboxes, gt_bboxes, self.num_buckets,
                                     self.scale_factor, self.offset_topk,
                                     self.offset_allow,
                                     self.cls_ignore_neighbor)
        return encoded_bboxes

    def decode(self, bboxes, pred_bboxes, max_shape=None):
        """Apply transformation `pred_bboxes` to `boxes`."""
        assert len(pred_bboxes) == 2
        cls_preds, offset_preds = pred_bboxes
        assert cls_preds.size(0) == bboxes.size(0) and offset_preds.size(
            0) == bboxes.size(0)
        decoded_bboxes = bucket2bbox(bboxes, cls_preds, offset_preds,
                                     self.num_buckets, self.scale_factor,
                                     max_shape)

        return decoded_bboxes


def generat_buckets(proposals, num_buckets, scale_factor=1.0):
    """Generate buckets w.r.t bucket number and scale factor of proposals.

    Args:
        proposals (Tensor): Shape (n, 4)
        num_buckets (int): Number of buckets.
        scale_factor (float): Scale factor to rescale proposals.

    Returns:
        tuple[Tensor]: (bucket_pw, bucket_ph, l_buckets, \
            r_buckets, t_buckets, d_buckets)

            - bucket_pw: Width of buckets on x-axis. Shape (n, ).
            - bucket_ph: Height of buckets on y-axis. Shape (n, ).
            - l_buckets: Left buckets. Shape (n, ceil(side_num/2)).
            - r_buckets: Right buckets. Shape (n, ceil(side_num/2)).
            - t_buckets: Top buckets. Shape (n, ceil(side_num/2)).
            - d_buckets: Down buckets. Shape (n, ceil(side_num/2)).
    """
    proposals = bbox_rescale(proposals, scale_factor)

    # number of buckets in each side
    side_num = int(np.ceil(num_buckets / 2.0))
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]
    px1 = proposals[..., 0]
    py1 = proposals[..., 1]
    px2 = proposals[..., 2]
    py2 = proposals[..., 3]

    bucket_pw = pw / num_buckets
    bucket_ph = ph / num_buckets

    # left buckets
    l_buckets = px1[:, None] + (0.5 + torch.arange(
        0, side_num).cuda().float())[None, :] * bucket_pw[:, None]
    # right buckets
    r_buckets = px2[:, None] - (0.5 + torch.arange(
        0, side_num).cuda().float())[None, :] * bucket_pw[:, None]
    # top buckets
    t_buckets = py1[:, None] + (0.5 + torch.arange(
        0, side_num).cuda().float())[None, :] * bucket_ph[:, None]
    # down buckets
    d_buckets = py2[:, None] - (0.5 + torch.arange(
        0, side_num).cuda().float())[None, :] * bucket_ph[:, None]
    return bucket_pw, bucket_ph, l_buckets, r_buckets, t_buckets, d_buckets


def label2onehot(labels, label_num):
    flat_labels = labels.new_zeros((labels.size(0), label_num)).float()
    flat_labels[torch.arange(0, labels.size(0)).cuda().long(), labels] = 1.0
    return flat_labels


def bbox2bucket(proposals,
                gt,
                num_buckets,
                scale_factor,
                offset_topk=2,
                offset_allow=1.0,
                cls_ignore_neighbor=True):
    """Generate buckets estimation and fine regression targets.

    Args:
        proposals (Tensor): Shape (n, 4)
        gt (Tensor): Shape (n, 4)
        num_buckets (int): Number of buckets.
        scale_factor (float): Scale factor to rescale proposals.
        offset_topk (int): Topk buckets are used to generate \
            bucket fine regression targets. Defaults to 2.
        offset_allow (float): Offset allowance to generate \
            bucket fine regression targets. \
            To avoid too large offset displacements. Defaults to 1.0.
        cls_ignore_neighbor (bool): Ignore second nearest bucket or Not. \
            Defaults to True.

    Returns:
        tuple[Tensor]: (offsets, offsets_weights, bucket_labels, cls_weights).

            - offsets: Fine regression targets.
                Shape (n, num_buckets*2).
            - offsets_weights: Fine regression weights.
                Shape (n, num_buckets*2).
            - bucket_labels: Bucketing estimation labels.
                Shape (n, num_buckets*2).
            - cls_weights: Bucketing estimation weights.
                Shape (n, num_buckets*2).
    """
    assert proposals.size() == gt.size()

    # generate buckets
    proposals = proposals.float()
    gt = gt.float()
    (bucket_pw, bucket_ph, l_buckets, r_buckets, t_buckets,
     d_buckets) = generat_buckets(proposals, num_buckets, scale_factor)

    gx1 = gt[..., 0]
    gy1 = gt[..., 1]
    gx2 = gt[..., 2]
    gy2 = gt[..., 3]

    # generate offset targets and weights
    # offsets from buckets to gts
    l_offsets = (l_buckets - gx1[:, None]) / bucket_pw[:, None]
    r_offsets = (r_buckets - gx2[:, None]) / bucket_pw[:, None]
    t_offsets = (t_buckets - gy1[:, None]) / bucket_ph[:, None]
    d_offsets = (d_buckets - gy2[:, None]) / bucket_ph[:, None]

    # select top-k nearset buckets
    l_topk, l_label = l_offsets.abs().topk(
        offset_topk, dim=1, largest=False, sorted=True)
    r_topk, r_label = r_offsets.abs().topk(
        offset_topk, dim=1, largest=False, sorted=True)
    t_topk, t_label = t_offsets.abs().topk(
        offset_topk, dim=1, largest=False, sorted=True)
    d_topk, d_label = d_offsets.abs().topk(
        offset_topk, dim=1, largest=False, sorted=True)

    offset_l_weights = l_offsets.new_zeros(l_offsets.size())
    offset_r_weights = r_offsets.new_zeros(r_offsets.size())
    offset_t_weights = t_offsets.new_zeros(t_offsets.size())
    offset_d_weights = d_offsets.new_zeros(d_offsets.size())
    inds = torch.arange(0, proposals.size(0)).cuda().long()

    # generate offset weights of top-k nearset buckets
    for k in range(offset_topk):
        if k >= 1:
            offset_l_weights[inds, l_label[:, k]] = (l_topk[:, k] <
                                                     offset_allow).float()
            offset_r_weights[inds, r_label[:, k]] = (r_topk[:, k] <
                                                     offset_allow).float()
            offset_t_weights[inds, t_label[:, k]] = (t_topk[:, k] <
                                                     offset_allow).float()
            offset_d_weights[inds, d_label[:, k]] = (d_topk[:, k] <
                                                     offset_allow).float()
        else:
            offset_l_weights[inds, l_label[:, k]] = 1.0
            offset_r_weights[inds, r_label[:, k]] = 1.0
            offset_t_weights[inds, t_label[:, k]] = 1.0
            offset_d_weights[inds, d_label[:, k]] = 1.0

    offsets = torch.cat([l_offsets, r_offsets, t_offsets, d_offsets], dim=-1)
    offsets_weights = torch.cat([
        offset_l_weights, offset_r_weights, offset_t_weights, offset_d_weights
    ],
                                dim=-1)

    # generate bucket labels and weight
    side_num = int(np.ceil(num_buckets / 2.0))
    labels = torch.cat([
        l_label[:, 0][:, None], r_label[:, 0][:, None], t_label[:, 0][:, None],
        d_label[:, 0][:, None]
    ],
                       dim=-1)

    batch_size = labels.size(0)
    bucket_labels = label2onehot(labels.view(-1),
                                 side_num).view(batch_size, -1)
    bucket_cls_l_weights = (l_offsets.abs() < 1).float()
    bucket_cls_r_weights = (r_offsets.abs() < 1).float()
    bucket_cls_t_weights = (t_offsets.abs() < 1).float()
    bucket_cls_d_weights = (d_offsets.abs() < 1).float()
    bucket_cls_weights = torch.cat([
        bucket_cls_l_weights, bucket_cls_r_weights, bucket_cls_t_weights,
        bucket_cls_d_weights
    ],
                                   dim=-1)
    # ignore second nearest buckets for cls if necessay
    if cls_ignore_neighbor:
        bucket_cls_weights = (~((bucket_cls_weights == 1) &
                                (bucket_labels == 0))).float()
    else:
        bucket_cls_weights[:] = 1.0
    return offsets, offsets_weights, bucket_labels, bucket_cls_weights


def bucket2bbox(proposals,
                cls_preds,
                offset_preds,
                num_buckets,
                scale_factor=1.0,
                max_shape=None):
    """Apply bucketing estimation (cls preds) and fine regression (offset
    preds) to generate det bboxes."""
    side_num = int(np.ceil(num_buckets / 2.0))
    cls_preds = cls_preds.view(-1, side_num)
    offset_preds = offset_preds.view(-1, side_num)

    scores = F.softmax(cls_preds, dim=1)
    score_topk, score_label = scores.topk(2, dim=1, largest=True, sorted=True)

    rescaled_proposals = bbox_rescale(proposals, scale_factor)

    pw = rescaled_proposals[..., 2] - rescaled_proposals[..., 0]
    ph = rescaled_proposals[..., 3] - rescaled_proposals[..., 1]
    px1 = rescaled_proposals[..., 0]
    py1 = rescaled_proposals[..., 1]
    px2 = rescaled_proposals[..., 2]
    py2 = rescaled_proposals[..., 3]

    bucket_pw = pw / num_buckets
    bucket_ph = ph / num_buckets

    score_inds_l = score_label[0::4, 0]
    score_inds_r = score_label[1::4, 0]
    score_inds_t = score_label[2::4, 0]
    score_inds_d = score_label[3::4, 0]
    l_buckets = px1 + (0.5 + score_inds_l.float()) * bucket_pw
    r_buckets = px2 - (0.5 + score_inds_r.float()) * bucket_pw
    t_buckets = py1 + (0.5 + score_inds_t.float()) * bucket_ph
    d_buckets = py2 - (0.5 + score_inds_d.float()) * bucket_ph

    offsets = offset_preds.view(-1, 4, side_num)
    inds = torch.arange(proposals.size(0)).cuda().long()
    l_offsets = offsets[:, 0, :][inds, score_inds_l]
    r_offsets = offsets[:, 1, :][inds, score_inds_r]
    t_offsets = offsets[:, 2, :][inds, score_inds_t]
    d_offsets = offsets[:, 3, :][inds, score_inds_d]

    x1 = l_buckets - l_offsets * bucket_pw
    x2 = r_buckets - r_offsets * bucket_pw
    y1 = t_buckets - t_offsets * bucket_ph
    y2 = d_buckets - d_offsets * bucket_ph

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.cat([x1[:, None], y1[:, None], x2[:, None], y2[:, None]],
                       dim=-1)

    # bucketing guided rescoring
    loc_confidence = score_topk[:, 0]
    top2_neighbor_inds = (score_label[:, 0] - score_label[:, 1]).abs() == 1
    loc_confidence += score_topk[:, 1] * top2_neighbor_inds.float()
    loc_confidence = loc_confidence.view(-1, 4).mean(dim=1)

    return bboxes, loc_confidence
