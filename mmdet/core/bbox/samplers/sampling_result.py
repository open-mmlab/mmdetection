import torch


class SamplingResult(object):#当完成正负样本采样以后，用这个类将正负样本的对应索引全部保存

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        #正样本个数
        self.pos_inds = pos_inds
        #负样本个数
        self.neg_inds = neg_inds
        #正样本
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
    def bboxes(self):#返回所有的正负样本
        return torch.cat([self.pos_bboxes, self.neg_bboxes])
