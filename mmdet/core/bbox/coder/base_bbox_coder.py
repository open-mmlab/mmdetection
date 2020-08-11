from abc import ABCMeta, abstractmethod


class BaseBBoxCoder(metaclass=ABCMeta):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
        pass

    @abstractmethod
    def decode(self, bboxes, bboxes_pred):
        pass
