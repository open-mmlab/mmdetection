from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class YOLOXModeSwitchHook(Hook):
    """Switch the mode of YOLOX during training.

    This hook turns off the mosaic and mixup data augmentation and switches
    to use L1 loss in bbox_head.

    Args:
        last_epoch (int): The last epoch of close data augmentation and
           switch loss. Default: 15.
    """

    def __init__(self, last_epoch=15):
        self.last_epoch = last_epoch

    def before_train_epoch(self, runner):
        """Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        train_loader = runner.data_loader
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if (epoch + 1) == runner.max_epochs - self.last_epoch:
            runner.logger.info('No mosaic and mixup aug now!')
            # TODO
            train_loader.dataset.enable_mosaic = False
            train_loader.dataset.enable_mixup = False
            runner.logger.info('Add additional L1 loss now!')
            model.bbox_head.use_l1 = True
