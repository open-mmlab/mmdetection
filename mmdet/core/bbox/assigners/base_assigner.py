from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):

    @abstractmethod
    def assign(self,
               bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               background_label=-1):
        pass
