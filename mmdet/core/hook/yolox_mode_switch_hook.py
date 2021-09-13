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
            skip pipeline. Default: ('Mosaic', 'RandomAffine', 'MixUp').
       is_multiprocess (bool): Determine whether it is a multi-process.
            If num_workers in the dataloader or workers_per_gpu in
            the configuration is greater than 0, it should be set to true.
            Default: True.
    """

    def __init__(self,
                 num_last_epochs=15,
                 skip_type_keys=('Mosaic', 'RandomAffine', 'MixUp'),
                 is_multiprocess=True):
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys
        self.is_multiprocess = is_multiprocess

    def before_train_epoch(self, runner):
        """Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        train_loader = runner.data_loader
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if (epoch + 1) == runner.max_epochs - self.num_last_epochs:
            runner.logger.info('Add additional L1 loss now!')
            model.bbox_head.use_l1 = True

        if self.is_multiprocess:
            # Due to the multi-process feature, closing the augmentation
            # operation requires an epoch in advance.
            closed_epoch = runner.max_epochs - self.num_last_epochs - 1
        else:
            closed_epoch = runner.max_epochs - self.num_last_epochs
        if (epoch + 1) == closed_epoch:
            runner.logger.info('No mosaic and mixup aug!')
            train_loader.dataset.update_skip_type_keys(self.skip_type_keys)
