import numpy as np
import torch

from .random_sampler import RandomSampler


class IoUBalancedNegSampler(RandomSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 hard_thr=0.1,
                 hard_fraction=0.5,
                 **kwargs):
        super(IoUBalancedNegSampler, self).__init__(num, pos_fraction,
                                                    **kwargs)
        assert hard_thr > 0
        assert 0 < hard_fraction < 1
        self.hard_thr = hard_thr
        self.hard_fraction = hard_fraction

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            max_overlaps = assign_result.max_overlaps.cpu().numpy()
            # balance sampling for negative samples
            neg_set = set(neg_inds.cpu().numpy())
            easy_set = set(
                np.where(
                    np.logical_and(max_overlaps >= 0,
                                   max_overlaps < self.hard_thr))[0])
            hard_set = set(np.where(max_overlaps >= self.hard_thr)[0])
            easy_neg_inds = list(easy_set & neg_set)
            hard_neg_inds = list(hard_set & neg_set)

            num_expected_hard = int(num_expected * self.hard_fraction)
            if len(hard_neg_inds) > num_expected_hard:
                sampled_hard_inds = self.random_choice(hard_neg_inds,
                                                       num_expected_hard)
            else:
                sampled_hard_inds = np.array(hard_neg_inds, dtype=np.int)
            num_expected_easy = num_expected - len(sampled_hard_inds)
            if len(easy_neg_inds) > num_expected_easy:
                sampled_easy_inds = self.random_choice(easy_neg_inds,
                                                       num_expected_easy)
            else:
                sampled_easy_inds = np.array(easy_neg_inds, dtype=np.int)
            sampled_inds = np.concatenate((sampled_easy_inds,
                                           sampled_hard_inds))
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(list(neg_set - set(sampled_inds)))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                sampled_inds = np.concatenate((sampled_inds, extra_inds))
            sampled_inds = torch.from_numpy(sampled_inds).long().to(
                assign_result.gt_inds.device)
            return sampled_inds
