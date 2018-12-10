import torch

from .base_sampler import BaseSampler
from ..transforms import bbox2roi
from .sampling_result import SamplingResult


class OHEMSampler(BaseSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,):
        super(OHEMSampler, self).__init__()
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals

    def _sample_pos(self, assign_result, num_expected, loss_all):
        """Hard sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            _, topk_loss_pos_inds = loss_all[pos_inds].topk(num_expected)
            return pos_inds[topk_loss_pos_inds]

    def _sample_neg(self, assign_result, num_expected, loss_all):
        """Hard sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            _, topk_loss_neg_inds = loss_all[neg_inds].topk(num_expected)
            return neg_inds[topk_loss_neg_inds]

    def sample(self, assign_result, bboxes, gt_bboxes, gt_labels=None,
               feats=None, bbox_roi_extractor=None, bbox_head=None):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        # calculate loss of all samples used for hard mining
        with torch.no_grad():
            rois = bbox2roi([bboxes])
            bbox_feats = bbox_roi_extractor(
                feats[:bbox_roi_extractor.num_inputs], rois)
            cls_score, _ = bbox_head(bbox_feats)
            loss_all = bbox_head.loss(
                cls_score=cls_score,
                bbox_pred=None,
                labels=assign_result.labels,
                label_weights=cls_score.new_ones(cls_score.size(0)),
                bbox_targets=None,
                bbox_weights=None,
                reduction='none')['loss_cls']

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self._sample_pos(assign_result, num_expected_pos, loss_all)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self._sample_neg(assign_result, num_expected_neg, loss_all)
        neg_inds = neg_inds.unique()

        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                              assign_result, gt_flags)
