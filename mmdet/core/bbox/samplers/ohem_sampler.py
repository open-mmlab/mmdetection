import torch

from .base_sampler import BaseSampler
from ..transforms import bbox2roi


class OHEMSampler(BaseSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 bbox_roi_extractor=None,
                 bbox_head=None):
        super(OHEMSampler, self).__init__()
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.bbox_roi_extractor = bbox_roi_extractor
        self.bbox_head = bbox_head

    def _sample_pos(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats=None,
                    **kwargs):
        """Hard sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            with torch.no_grad():
                rois = bbox2roi([bboxes[pos_inds]])
                bbox_feats = self.bbox_roi_extractor(
                    feats[:self.bbox_roi_extractor.num_inputs], rois)
                cls_score, _ = self.bbox_head(bbox_feats)
                loss_pos = self.bbox_head.loss(
                    cls_score=cls_score,
                    bbox_pred=None,
                    labels=assign_result.labels[pos_inds],
                    label_weights=cls_score.new_ones(cls_score.size(0)),
                    bbox_targets=None,
                    bbox_weights=None,
                    reduce=False)['loss_cls']
                _, topk_loss_pos_inds = loss_pos.topk(num_expected)
            return pos_inds[topk_loss_pos_inds]

    def _sample_neg(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats=None,
                    **kwargs):
        """Hard sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            with torch.no_grad():
                rois = bbox2roi([bboxes[neg_inds]])
                bbox_feats = self.bbox_roi_extractor(
                    feats[:self.bbox_roi_extractor.num_inputs], rois)
                cls_score, _ = self.bbox_head(bbox_feats)
                loss_neg = self.bbox_head.loss(
                    cls_score=cls_score,
                    bbox_pred=None,
                    labels=assign_result.labels[neg_inds],
                    label_weights=cls_score.new_ones(cls_score.size(0)),
                    bbox_targets=None,
                    bbox_weights=None,
                    reduce=False)['loss_cls']
                _, topk_loss_neg_inds = loss_neg.topk(num_expected)
            return neg_inds[topk_loss_neg_inds]
