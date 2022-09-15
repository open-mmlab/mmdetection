# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig


@MODELS.register_module()
class BasePanopticFusionHead(BaseModule, metaclass=ABCMeta):
    """Base class for panoptic heads."""

    def __init__(self,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 test_cfg: OptConfigType = None,
                 loss_panoptic: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_things_classes + num_stuff_classes
        self.test_cfg = test_cfg

        if loss_panoptic:
            self.loss_panoptic = MODELS.build(loss_panoptic)
        else:
            self.loss_panoptic = None

    @property
    def with_loss(self) -> bool:
        """bool: whether the panoptic head contains loss function."""
        return self.loss_panoptic is not None

    @abstractmethod
    def loss(self, **kwargs):
        """Loss function."""

    @abstractmethod
    def predict(self, **kwargs):
        """Predict function."""
