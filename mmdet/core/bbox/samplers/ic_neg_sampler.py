import torch
from ..builder import BBOX_SAMPLERS
from .ic_base_sampler import ICBaseSampler

@BBOX_SAMPLERS.register_module()
class ICNegSampler(ICBaseSampler):
    def __init__(self, num, pos_fraction,

                 **kwargs):
        super(ICNegSampler, self).__init__(num, pos_fraction,
                                           **kwargs)

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

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
            gallery = torch.tensor(
                gallery, dtype=torch.long, device=torch.cuda.current_device())
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def weight_random_choice(self, prob, gallery, num):
        assert len(gallery) >= num and len(gallery) == len(prob)
        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            gallery = torch.tensor(
                gallery, dtype=torch.long, device=torch.cuda.current_device())

        perm = torch.multinomial(prob, num_samples=num, replacement=False)
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def cal_weight_samples(self, iou, conf):
        lam = 0.6
        neg_scores = torch.exp(lam*iou + (1-lam)*conf)

        return neg_scores

    def _sample_neg(self, assign_result, num_expected, conf, iod, **kwargs):
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)

        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            neg_confs = conf[neg_inds]
            neg_iods = iod[neg_inds].detach()
           # neg_ious = assign_result.max_overlaps[neg_inds].detach()
            neg_probs = self.cal_weight_samples(neg_iods, neg_confs)
            return self.weight_random_choice(neg_probs, neg_inds, num_expected)

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)

        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)

        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)
