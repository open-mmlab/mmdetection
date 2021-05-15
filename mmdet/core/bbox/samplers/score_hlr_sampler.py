import torch
from mmcv.ops import nms_match

from ..builder import BBOX_SAMPLERS
from ..transforms import bbox2roi
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


@BBOX_SAMPLERS.register_module()
class ScoreHLRSampler(BaseSampler):
    r"""Importance-based Sample Reweighting (ISR_N), described in `Prime Sample
    Attention in Object Detection <https://arxiv.org/abs/1904.04821>`_.

    Score hierarchical local rank (HLR) differentiates with RandomSampler in
    negative part. It firstly computes Score-HLR in a two-step way,
    then linearly maps score hlr to the loss weights.

    Args:
        num (int): Total number of sampled RoIs.
        pos_fraction (float): Fraction of positive samples.
        context (:class:`BaseRoIHead`): RoI head that the sampler belongs to.
        neg_pos_ub (int): Upper bound of the ratio of num negative to num
            positive, -1 means no upper bound.
        add_gt_as_proposals (bool): Whether to add ground truth as proposals.
        k (float): Power of the non-linear mapping.
        bias (float): Shift of the non-linear mapping.
        score_thr (float): Minimum score that a negative sample is to be
            considered as valid bbox.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 context,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 k=0.5,
                 bias=0,
                 score_thr=0.05,
                 iou_thr=0.5,
                 **kwargs):
        super().__init__(num, pos_fraction, neg_pos_ub, add_gt_as_proposals)
        self.k = k
        self.bias = bias
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.context = context
        # context of cascade detectors is a list, so distinguish them here.
        if not hasattr(context, 'num_stages'):
            self.bbox_roi_extractor = context.bbox_roi_extractor
            self.bbox_head = context.bbox_head
            self.with_shared_head = context.with_shared_head
            if self.with_shared_head:
                self.shared_head = context.shared_head
        else:
            self.bbox_roi_extractor = context.bbox_roi_extractor[
                context.current_stage]
            self.bbox_head = context.bbox_head[context.current_stage]

    @staticmethod
    def random_choice(gallery, num):
        """Randomly select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0).flatten()
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self,
                    assign_result,
                    num_expected,
                    bboxes,
                    feats=None,
                    img_meta=None,
                    **kwargs):
        """Sample negative samples.

        Score-HLR sampler is done in the following steps:
        1. Take the maximum positive score prediction of each negative samples
            as s_i.
        2. Filter out negative samples whose s_i <= score_thr, the left samples
            are called valid samples.
        3. Use NMS-Match to divide valid samples into different groups,
            samples in the same group will greatly overlap with each other
        4. Rank the matched samples in two-steps to get Score-HLR.
            (1) In the same group, rank samples with their scores.
            (2) In the same score rank across different groups,
                rank samples with their scores again.
        5. Linearly map Score-HLR to the final label weights.

        Args:
            assign_result (:obj:`AssignResult`): result of assigner.
            num_expected (int): Expected number of samples.
            bboxes (Tensor): bbox to be sampled.
            feats (Tensor): Features come from FPN.
            img_meta (dict): Meta information dictionary.
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0).flatten()
        num_neg = neg_inds.size(0)
        if num_neg == 0:
            return neg_inds, None
        with torch.no_grad():
            neg_bboxes = bboxes[neg_inds]
            neg_rois = bbox2roi([neg_bboxes])
            bbox_result = self.context._bbox_forward(feats, neg_rois)
            cls_score, bbox_pred = bbox_result['cls_score'], bbox_result[
                'bbox_pred']

            ori_loss = self.bbox_head.loss(
                cls_score=cls_score,
                bbox_pred=None,
                rois=None,
                labels=neg_inds.new_full((num_neg, ),
                                         self.bbox_head.num_classes),
                label_weights=cls_score.new_ones(num_neg),
                bbox_targets=None,
                bbox_weights=None,
                reduction_override='none')['loss_cls']

            # filter out samples with the max score lower than score_thr
            max_score, argmax_score = cls_score.softmax(-1)[:, :-1].max(-1)
            valid_inds = (max_score > self.score_thr).nonzero().view(-1)
            invalid_inds = (max_score <= self.score_thr).nonzero().view(-1)
            num_valid = valid_inds.size(0)
            num_invalid = invalid_inds.size(0)

            num_expected = min(num_neg, num_expected)
            num_hlr = min(num_valid, num_expected)
            num_rand = num_expected - num_hlr
            if num_valid > 0:
                valid_rois = neg_rois[valid_inds]
                valid_max_score = max_score[valid_inds]
                valid_argmax_score = argmax_score[valid_inds]
                valid_bbox_pred = bbox_pred[valid_inds]

                # valid_bbox_pred shape: [num_valid, #num_classes, 4]
                valid_bbox_pred = valid_bbox_pred.view(
                    valid_bbox_pred.size(0), -1, 4)
                selected_bbox_pred = valid_bbox_pred[range(num_valid),
                                                     valid_argmax_score]
                pred_bboxes = self.bbox_head.bbox_coder.decode(
                    valid_rois[:, 1:], selected_bbox_pred)
                pred_bboxes_with_score = torch.cat(
                    [pred_bboxes, valid_max_score[:, None]], -1)
                group = nms_match(pred_bboxes_with_score, self.iou_thr)

                # imp: importance
                imp = cls_score.new_zeros(num_valid)
                for g in group:
                    g_score = valid_max_score[g]
                    # g_score has already sorted
                    rank = g_score.new_tensor(range(g_score.size(0)))
                    imp[g] = num_valid - rank + g_score
                _, imp_rank_inds = imp.sort(descending=True)
                _, imp_rank = imp_rank_inds.sort()
                hlr_inds = imp_rank_inds[:num_expected]

                if num_rand > 0:
                    rand_inds = torch.randperm(num_invalid)[:num_rand]
                    select_inds = torch.cat(
                        [valid_inds[hlr_inds], invalid_inds[rand_inds]])
                else:
                    select_inds = valid_inds[hlr_inds]

                neg_label_weights = cls_score.new_ones(num_expected)

                up_bound = max(num_expected, num_valid)
                imp_weights = (up_bound -
                               imp_rank[hlr_inds].float()) / up_bound
                neg_label_weights[:num_hlr] = imp_weights
                neg_label_weights[num_hlr:] = imp_weights.min()
                neg_label_weights = (self.bias +
                                     (1 - self.bias) * neg_label_weights).pow(
                                         self.k)
                ori_selected_loss = ori_loss[select_inds]
                new_loss = ori_selected_loss * neg_label_weights
                norm_ratio = ori_selected_loss.sum() / new_loss.sum()
                neg_label_weights *= norm_ratio
            else:
                neg_label_weights = cls_score.new_ones(num_expected)
                select_inds = torch.randperm(num_neg)[:num_expected]

            return neg_inds[select_inds], neg_label_weights

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               img_meta=None,
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
            tuple[:obj:`SamplingResult`, Tensor]: Sampling result and negative
                label weights.
        """
        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds, neg_label_weights = self.neg_sampler._sample_neg(
            assign_result,
            num_expected_neg,
            bboxes,
            img_meta=img_meta,
            **kwargs)

        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                              assign_result, gt_flags), neg_label_weights
