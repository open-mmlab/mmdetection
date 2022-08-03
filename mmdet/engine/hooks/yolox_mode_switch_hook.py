# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class YOLOXModeSwitchHook(Hook):
    """Switch the mode of YOLOX during training.

    This hook turns off the mosaic and mixup data augmentation and switches
    to use L1 loss in bbox_head.

    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to close the data augmentation and switch to L1 loss.
            Defaults to 15.
       skip_type_keys (Sequence[str], optional): Sequence of type string to be
            skip pipeline. Defaults to ('Mosaic', 'RandomAffine', 'MixUp').
    """

    def __init__(
        self,
        num_last_epochs: int = 15,
        skip_type_keys: Sequence[str] = ('Mosaic', 'RandomAffine', 'MixUp')
    ) -> None:
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys

    def before_train_epoch(self, runner) -> None:
        """Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if (epoch + 1) == runner.max_epochs - self.num_last_epochs:
            runner.logger.info('Add additional L1 loss now!')
            model.bbox_head.use_l1 = True
