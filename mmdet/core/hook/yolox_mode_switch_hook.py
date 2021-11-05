# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class YOLOXModeSwitchHook(Hook):
    """Switch the mode of YOLOX during training.

    This hook turns off the mosaic and mixup data augmentation and switches
    to use L1 loss in bbox_head.

    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to close the data augmentation and switch to L1 loss.
            Default: 15.
       skip_type_keys (list[str], optional): Sequence of type string to be
            skip pipeline. Default: ('Mosaic', 'RandomAffine', 'MixUp')
    """

    def __init__(self,
                 num_last_epochs=15,
                 skip_type_keys=('Mosaic', 'RandomAffine', 'MixUp')):
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys
        self.reset_persistent_workers = False

    def before_train_epoch(self, runner):
        """Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        train_loader = runner.data_loader
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if (epoch + 1) == runner.max_epochs - self.num_last_epochs:
            runner.logger.info('No mosaic and mixup aug now!')
            train_loader.dataset.update_skip_type_keys(self.skip_type_keys)
            if hasattr(train_loader, 'persistent_workers') and train_loader.persistent_workers is True:
                self.reset_persistent_workers = True
                train_loader.persistent_workers = False
            runner.logger.info('Add additional L1 loss now!')
            model.bbox_head.use_l1 = True

    def after_train_epoch(self, runner):
        train_loader = runner.data_loader
        if hasattr(train_loader, 'persistent_workers') and self.reset_persistent_workers:
            if train_loader.persistent_workers is False:
                train_loader.persistent_workers = True
