import sys
from abc import ABCMeta, abstractmethod

import torch.nn as nn

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


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
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        outs = self.__call__(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    if sys.version_info >= (3, 7):

        async def async_simple_test(self, x, img_metas):
            sleep_interval = self.test_cfg.pop('async_sleep_interval', 0.025)
            async with completed(
                    __name__, 'dense_head_forward',
                    sleep_interval=sleep_interval):
                outs = self.__call__(x)

            bboxes = self.get_bboxes(*outs, img_metas)
            return bboxes

    def simple_test(self, x, img_metas, rescale=False):
        outs = self.__call__(x)
        bboxes = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return bboxes

    def aug_test(self, feats, img_metas):
        raise NotImplementedError
