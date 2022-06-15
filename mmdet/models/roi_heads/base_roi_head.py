# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Tuple

from mmengine.model import BaseModule
from torch import Tensor

from mmdet.core.utils import (InstanceList, OptConfigType, OptMultiConfig,
                              SampleList)
from mmdet.registry import MODELS


class BaseRoIHead(BaseModule, metaclass=ABCMeta):
    """Base class for RoIHeads."""

    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            self.shared_head = MODELS.build(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        self.init_assigner_sampler()

    @property
    def with_bbox(self) -> bool:
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self) -> bool:
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_shared_head(self) -> bool:
        """bool: whether the RoI head contains a `shared_head`"""
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @abstractmethod
    def init_bbox_head(self, *args, **kwargs):
        """Initialize ``bbox_head``"""
        pass

    @abstractmethod
    def init_mask_head(self, *args, **kwargs):
        """Initialize ``mask_head``"""
        pass

    @abstractmethod
    def init_assigner_sampler(self, *args, **kwargs):
        """Initialize assigner and sampler."""
        pass

    @abstractmethod
    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList, **kwargs):
        """Perform forward propagation and loss calculation of the roi head on
        the features of the upstream network."""

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False,
                **kwargs):
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network."""

    # TODO: Currently not supported
    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test function with test time augmentation.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
