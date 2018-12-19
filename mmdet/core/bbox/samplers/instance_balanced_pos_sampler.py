import numpy as np
import torch

from .random_sampler import RandomSampler


class InstanceBalancedPosSampler(RandomSampler):

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            unique_gt_inds = assign_result.gt_inds[pos_inds].unique()
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
                    inds = self.random_choice(inds, num_per_gt)
                sampled_inds.append(inds)
            sampled_inds = torch.cat(sampled_inds)
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(
                    list(set(pos_inds.cpu()) - set(sampled_inds.cpu())))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                extra_inds = torch.from_numpy(extra_inds).to(
                    assign_result.gt_inds.device).long()
                sampled_inds = torch.cat([sampled_inds, extra_inds])
            elif len(sampled_inds) > num_expected:
                sampled_inds = self.random_choice(sampled_inds, num_expected)
            return sampled_inds
