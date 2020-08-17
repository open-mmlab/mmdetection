import numpy as np
import torch
import torch.nn.functional as F

from ..transforms import bbox_rescale
from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class BucketingBBoxCoder(BaseBBoxCoder):
    """Delta XYWH BBox coder.
    """

    def __init__(self,
                 bucket_num,
                 scale_factor,
                 offset_topk=2,
                 offset_allow=1.0,
                 cls_ignore_neighbor=True):
        super(BucketingBBoxCoder, self).__init__()
        self.bucket_num = bucket_num
        self.scale_factor = scale_factor
        self.offset_topk = offset_topk
        self.offset_allow = offset_allow
        self.cls_ignore_neighbor = cls_ignore_neighbor

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = bbox2bucket(bboxes, gt_bboxes, self.bucket_num,
                                     self.scale_factor, self.offset_topk,
                                     self.offset_allow,
                                     self.cls_ignore_neighbor)
        return encoded_bboxes

    def decode(self, bboxes, pred_bboxes, max_shape=None):
        """Apply transformation `pred_bboxes` to `boxes`.
        """
        assert len(pred_bboxes) == 2
        cls_preds, offset_preds = pred_bboxes
        assert cls_preds.size(0) == bboxes.size(0) and offset_preds.size(
            0) == bboxes.size(0)
        decoded_bboxes = bucket2bbox(bboxes, cls_preds, offset_preds,
                                     self.bucket_num, self.scale_factor,
                                     max_shape)

        return decoded_bboxes


def generat_buckets(proposals, bucket_num, scale_factor=1.0):

    proposals = bbox_rescale(proposals, scale_factor)

    # number of buckets in each side
    side_num = int(np.ceil(bucket_num / 2.0))
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]
    px1 = proposals[..., 0]
    py1 = proposals[..., 1]
    px2 = proposals[..., 2]
    py2 = proposals[..., 3]

    edge_pw = pw / bucket_num
    edge_ph = ph / bucket_num

    # left edges
    l_edges = px1[:, None] + (0.5 + torch.arange(
        0, side_num).cuda().float())[None, :] * edge_pw[:, None]
    # right edges
    r_edges = px2[:, None] - (0.5 + torch.arange(
        0, side_num).cuda().float())[None, :] * edge_pw[:, None]
    # top edges
    t_edges = py1[:, None] + (0.5 + torch.arange(
        0, side_num).cuda().float())[None, :] * edge_ph[:, None]
    # down edges
    d_edges = py2[:, None] - (0.5 + torch.arange(
        0, side_num).cuda().float())[None, :] * edge_ph[:, None]
    return edge_pw, edge_ph, l_edges, r_edges, t_edges, d_edges


def label2onehot(labels, label_num):
    flat_labels = labels.new_zeros((labels.size(0), label_num)).float()
    flat_labels[torch.arange(0, labels.size(0)).cuda().long(), labels] = 1.0
    return flat_labels


def bbox2bucket(proposals,
                gt,
                bucket_num,
                scale_factor,
                offset_topk=2,
                offset_allow=1.0,
                cls_ignore_neighbor=True):
    """Compute deltas of proposals w.r.t. gt.
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    (edge_pw, edge_ph, l_edges, r_edges, t_edges,
     d_edges) = generat_buckets(proposals, bucket_num, scale_factor)

    gx1 = gt[..., 0]
    gy1 = gt[..., 1]
    gx2 = gt[..., 2]
    gy2 = gt[..., 3]

    l_offsets = (l_edges - gx1[:, None]) / edge_pw[:, None]
    r_offsets = (r_edges - gx2[:, None]) / edge_pw[:, None]
    t_offsets = (t_edges - gy1[:, None]) / edge_ph[:, None]
    d_offsets = (d_edges - gy2[:, None]) / edge_ph[:, None]

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

    side_num = int(np.ceil(bucket_num / 2.0))
    labels = torch.cat([
        l_label[:, 0][:, None], r_label[:, 0][:, None], t_label[:, 0][:, None],
        d_label[:, 0][:, None]
    ],
                       dim=-1)

    batch_size = labels.size(0)
    one_hot_labels = label2onehot(labels.view(-1),
                                  side_num).view(batch_size, -1)
    cls_l_weights = (l_offsets.abs() < 1).float()
    cls_r_weights = (r_offsets.abs() < 1).float()
    cls_t_weights = (t_offsets.abs() < 1).float()
    cls_d_weights = (d_offsets.abs() < 1).float()
    cls_weights = torch.cat(
        [cls_l_weights, cls_r_weights, cls_t_weights, cls_d_weights], dim=-1)
    if cls_ignore_neighbor:
        cls_weights = (~((cls_weights == 1) & (one_hot_labels == 0))).float()
    else:
        cls_weights[:] = 1.0
    return offsets, offsets_weights, one_hot_labels, cls_weights


def bucket2bbox(proposals,
                cls_preds,
                offset_preds,
                bucket_num,
                scale_factor=1.0,
                max_shape=None):
    """Apply deltas to shift/scale base boxes.
    """
    side_num = int(np.ceil(bucket_num / 2.0))
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

    edge_pw = pw / bucket_num
    edge_ph = ph / bucket_num

    score_inds_l = score_label[0::4, 0]
    score_inds_r = score_label[1::4, 0]
    score_inds_t = score_label[2::4, 0]
    score_inds_d = score_label[3::4, 0]
    l_edges = px1 + (0.5 + score_inds_l.float()) * edge_pw
    r_edges = px2 - (0.5 + score_inds_r.float()) * edge_pw
    t_edges = py1 + (0.5 + score_inds_t.float()) * edge_ph
    d_edges = py2 - (0.5 + score_inds_d.float()) * edge_ph

    offsets = offset_preds.view(-1, 4, side_num)
    inds = torch.arange(proposals.size(0)).cuda().long()
    l_offsets = offsets[:, 0, :][inds, score_inds_l]
    r_offsets = offsets[:, 1, :][inds, score_inds_r]
    t_offsets = offsets[:, 2, :][inds, score_inds_t]
    d_offsets = offsets[:, 3, :][inds, score_inds_d]

    x1 = l_edges - l_offsets * edge_pw
    x2 = r_edges - r_offsets * edge_pw
    y1 = t_edges - t_offsets * edge_ph
    y2 = d_edges - d_offsets * edge_ph

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.cat([x1[:, None], y1[:, None], x2[:, None], y2[:, None]],
                       dim=-1)

    loc_confidence = score_topk[:, 0]
    top2_neighbor_inds = (score_label[:, 0] - score_label[:, 1]).abs() == 1
    loc_confidence += score_topk[:, 1] * top2_neighbor_inds.float()
    loc_confidence = loc_confidence.view(-1, 4).mean(dim=1)

    return bboxes, loc_confidence
