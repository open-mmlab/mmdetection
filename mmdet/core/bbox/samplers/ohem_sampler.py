import torch

from ..builder import BBOX_SAMPLERS
from ..transforms import bbox2roi
from .base_sampler import BaseSampler


@BBOX_SAMPLERS.register_module()
class OHEMSampler(BaseSampler):
    """
    Online Hard Example Mining Sampler described in [1]_.

    References:
        .. [1] https://arxiv.org/pdf/1604.03540.pdf
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 context,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        super(OHEMSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                          add_gt_as_proposals)
        if not hasattr(context, 'num_stages'):
            self.bbox_roi_extractor = context.bbox_roi_extractor
            self.bbox_head = context.bbox_head
        else:
            self.bbox_roi_extractor = context.bbox_roi_extractor[
                context.current_stage]
            self.bbox_head = context.bbox_head[context.current_stage]

    def hard_mining(self, inds, num_expected, bboxes, labels, feats):
        with torch.no_grad():
            rois = bbox2roi([bboxes])
            bbox_feats = self.bbox_roi_extractor(
                feats[:self.bbox_roi_extractor.num_inputs], rois)
            cls_score, _ = self.bbox_head(bbox_feats)
            loss = self.bbox_head.loss(
                cls_score=cls_score,
                bbox_pred=None,
                rois=rois,
                labels=labels,
                label_weights=cls_score.new_ones(cls_score.size(0)),
                bbox_targets=None,
                bbox_weights=None,
                reduction_override='none')['loss_cls']
            _, topk_loss_inds = loss.topk(num_expected)
        return inds[topk_loss_inds]

    def _sample_pos(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats=None,
                    **kwargs):
        """Sample positive boxes

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            num_expected (int): Number of expected positive samples
            bboxes (torch.Tensor, optional): Boxes. Defaults to None.
            feats (list[torch.Tensor], optional): Multi-level features.
                Defaults to None.

        Returns:
            torch.Tensor: Indices  of positive samples
        """
        # Sample some hard positive samples
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.hard_mining(pos_inds, num_expected, bboxes[pos_inds],
                                    assign_result.labels[pos_inds], feats)

    def _sample_neg(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats=None,
                    **kwargs):
        """Sample negative boxes

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            num_expected (int): Number of expected negative samples
            bboxes (torch.Tensor, optional): Boxes. Defaults to None.
            feats (list[torch.Tensor], optional): Multi-level features.
                Defaults to None.

        Returns:
            torch.Tensor: Indices  of negative samples
        """
        # Sample some hard negative samples
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            neg_labels = assign_result.labels.new_empty(
                neg_inds.size(0)).fill_(self.bbox_head.num_classes)
            return self.hard_mining(neg_inds, num_expected, bboxes[neg_inds],
                                    neg_labels, feats)
