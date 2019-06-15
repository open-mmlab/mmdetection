import torch
import torch.nn.functional as F

from .sampling_result import SamplingResult
from ..transforms import bbox2roi, bbox2delta


class RFCNOHEMSampler:

    def __init__(self,
                 num,
                 pos_fraction,
                 context,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.bbox_head = context.bbox_head
        self.cls_roi_extractor = context.cls_roi_extractor
        self.reg_roi_extractor = context.reg_roi_extractor

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               feats=None,
               **kwargs):
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

        with torch.no_grad():
            rois = bbox2roi([bboxes])
            cls_score, bbox_pred = self.bbox_head(feats, rois,
                                                  self.cls_roi_extractor,
                                                  self.reg_roi_extractor)
            target_bboxes = gt_bboxes[assign_result.gt_inds - 1]
            bbox_target = bbox2delta(
                bboxes,
                target_bboxes,
                means=self.bbox_head.target_means,
                stds=self.bbox_head.target_stds)
            pos_inds = torch.nonzero(assign_result.gt_inds > 0).view(-1)
            loss_cls = F.cross_entropy(
                cls_score, assign_result.labels, reduction='none')
            loss_bbox = F.smooth_l1_loss(
                bbox_pred, bbox_target, reduction='none').sum(-1)
            pos_loss_bbox = loss_bbox[pos_inds]
            loss = loss_cls.clone()
            loss[pos_inds] += pos_loss_bbox
            _, topk_loss_inds = loss.sort(descending=True)
            topk_loss_inds = topk_loss_inds[:self.num]

            # OHEM select topk (loss_cls + loss_reg) samples
            # If a pos sample in topk, then its mask is 3
            # If a neg sample in topk, then its mask is 1
            sample_mask = loss.new_zeros(loss.size(0)).byte()
            sample_mask[topk_loss_inds] += 1
            sample_mask[pos_inds] += 2

            pos_inds = (sample_mask == 3).nonzero().view(-1).unique()
            neg_inds = (sample_mask == 1).nonzero().view(-1).unique()

        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                              assign_result, gt_flags)
