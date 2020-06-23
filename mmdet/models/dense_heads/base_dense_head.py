from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseDenseHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads"""

    def __init__(self):
        super(BaseDenseHead, self).__init__()

    @abstractmethod
    def loss(self, **kwargs):
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
