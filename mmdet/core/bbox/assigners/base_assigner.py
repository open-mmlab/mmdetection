# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def assign(self,
               pred_instances,
               gt_instances,
               gt_instances_ignore=None,
               **kwargs):
        """Assign boxes to either a ground truth boxes or a negative boxes."""
