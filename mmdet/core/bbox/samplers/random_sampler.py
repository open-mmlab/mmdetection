import numpy as np
import torch

from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)

    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.
         随机选num个
        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num  #个数要足够
        if isinstance(gallery, list): #如果是个列表，转化成numpy
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))#所有的索引
        np.random.shuffle(cands)#打乱顺序
        rand_inds = cands[:num]#取出前num个
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)#重新转为tensor再放到设备上
        return gallery[rand_inds]

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)#正样本就是>0的样本
        if pos_inds.numel() != 0:#如果有
            pos_inds = pos_inds.squeeze(1)#去掉一维
        if pos_inds.numel() <= num_expected: #如果不够
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)
