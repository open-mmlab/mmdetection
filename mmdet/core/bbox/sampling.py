import numpy as np
import torch

from .assignment import BBoxAssigner


def random_choice(gallery, num):
    """Random select some elements from the gallery.

    It seems that Pytorch's implementation is slower than numpy so we use numpy
    to randperm the indices.
    """
    assert len(gallery) >= num
    if isinstance(gallery, list):
        gallery = np.array(gallery)
    cands = np.arange(len(gallery))
    np.random.shuffle(cands)
    rand_inds = cands[:num]
    if not isinstance(gallery, np.ndarray):
        rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
    return gallery[rand_inds]


def assign_and_sample(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels, cfg):
    bbox_assigner = BBoxAssigner(**cfg.assigner)
    bbox_sampler = BBoxSampler(**cfg.sampler)
    assign_result = bbox_assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore,
                                         gt_labels)
    sampling_result = bbox_sampler.sample(assign_result, bboxes, gt_bboxes,
                                          gt_labels)
    return assign_result, sampling_result


class BBoxSampler(object):
    """Sample positive and negative bboxes given assigned results.

    Args:
        pos_fraction (float): Positive sample fraction.
        neg_pos_ub (float): Negative/Positive upper bound.
        pos_balance_sampling (bool): Whether to sample positive samples around
            each gt bbox evenly.
        neg_balance_thr (float, optional): IoU threshold for simple/hard
            negative balance sampling.
        neg_hard_fraction (float, optional): Fraction of hard negative samples
            for negative balance sampling.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 pos_balance_sampling=False,
                 neg_balance_thr=0,
                 neg_hard_fraction=0.5):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_balance_sampling = pos_balance_sampling
        self.neg_balance_thr = neg_balance_thr
        self.neg_hard_fraction = neg_hard_fraction

    def _sample_pos(self, assign_result, num_expected):
        """Balance sampling for positive bboxes/anchors.

        1. calculate average positive num for each gt: num_per_gt
        2. sample at most num_per_gt positives for each gt
        3. random sampling from rest anchors if not enough fg
        """
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        elif not self.pos_balance_sampling:
            return random_choice(pos_inds, num_expected)
        else:
            unique_gt_inds = torch.unique(
                assign_result.gt_inds[pos_inds].cpu())
            num_gts = len(unique_gt_inds)
            num_per_gt = int(round(num_expected / float(num_gts)) + 1)
            sampled_inds = []
            for i in unique_gt_inds:
                inds = torch.nonzero(assign_result.gt_inds == i.item())
                if inds.numel() != 0:
                    inds = inds.squeeze(1)
                else:
                    continue
                if len(inds) > num_per_gt:
                    inds = random_choice(inds, num_per_gt)
                sampled_inds.append(inds)
            sampled_inds = torch.cat(sampled_inds)
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(
                    list(set(pos_inds.cpu()) - set(sampled_inds.cpu())))
                if len(extra_inds) > num_extra:
                    extra_inds = random_choice(extra_inds, num_extra)
                extra_inds = torch.from_numpy(extra_inds).to(
                    assign_result.gt_inds.device).long()
                sampled_inds = torch.cat([sampled_inds, extra_inds])
            elif len(sampled_inds) > num_expected:
                sampled_inds = random_choice(sampled_inds, num_expected)
            return sampled_inds

    def _sample_neg(self, assign_result, num_expected):
        """Balance sampling for negative bboxes/anchors.

        Negative samples are split into 2 set: hard (balance_thr <= iou <
        neg_iou_thr) and easy (iou < balance_thr). The sampling ratio is
        controlled by `hard_fraction`.
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        elif self.neg_balance_thr <= 0:
            # uniform sampling among all negative samples
            return random_choice(neg_inds, num_expected)
        else:
            max_overlaps = assign_result.max_overlaps.cpu().numpy()
            # balance sampling for negative samples
            neg_set = set(neg_inds.cpu().numpy())
            easy_set = set(
                np.where(
                    np.logical_and(max_overlaps >= 0,
                                   max_overlaps < self.neg_balance_thr))[0])
            hard_set = set(np.where(max_overlaps >= self.neg_balance_thr)[0])
            easy_neg_inds = list(easy_set & neg_set)
            hard_neg_inds = list(hard_set & neg_set)

            num_expected_hard = int(num_expected * self.neg_hard_fraction)
            if len(hard_neg_inds) > num_expected_hard:
                sampled_hard_inds = random_choice(hard_neg_inds,
                                                  num_expected_hard)
            else:
                sampled_hard_inds = np.array(hard_neg_inds, dtype=np.int)
            num_expected_easy = num_expected - len(sampled_hard_inds)
            if len(easy_neg_inds) > num_expected_easy:
                sampled_easy_inds = random_choice(easy_neg_inds,
                                                  num_expected_easy)
            else:
                sampled_easy_inds = np.array(easy_neg_inds, dtype=np.int)
            sampled_inds = np.concatenate((sampled_easy_inds,
                                           sampled_hard_inds))
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(list(neg_set - set(sampled_inds)))
                if len(extra_inds) > num_extra:
                    extra_inds = random_choice(extra_inds, num_extra)
                sampled_inds = np.concatenate((sampled_inds, extra_inds))
            sampled_inds = torch.from_numpy(sampled_inds).long().to(
                assign_result.gt_inds.device)
            return sampled_inds

    def sample(self, assign_result, bboxes, gt_bboxes, gt_labels=None):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        1. Assign gt to each bbox.
        2. Add gt bboxes to the sampling pool (optional).
        3. Perform positive and negative sampling.

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
            gt_flags = torch.cat([
                bboxes.new_ones((gt_bboxes.shape[0], ), dtype=torch.uint8),
                gt_flags
            ])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self._sample_pos(assign_result, num_expected_pos)
        # We found that sampled indices have duplicated items occasionally.
        # (mab be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            num_neg_max = int(self.neg_pos_ub *
                              num_sampled_pos) if num_sampled_pos > 0 else int(
                                  self.neg_pos_ub)
            num_expected_neg = min(num_neg_max, num_expected_neg)
        neg_inds = self._sample_neg(assign_result, num_expected_neg)
        neg_inds = neg_inds.unique()

        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                              assign_result, gt_flags)


class SamplingResult(object):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])
