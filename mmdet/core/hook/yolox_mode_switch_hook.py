from mmcv.runner.hooks import HOOKS, Hook
from mmdet.utils import get_root_logger


@HOOKS.register_module()
class YOLOXModeSwitchHook(Hook):
    """Switch the model of YOLOX during training.

    This hook turns off the mosaic and mixup data augmentation and switches
    to use L1 loss in bbox_head.

    Args:
        switch_epoch (int): The epoch of close data augmentation. Default 15.
    """
    def __init__(self, switch_epoch=15):
        self.switch_epoch = switch_epoch

    def before_train_epoch(self, runner):
        """Close mosaic and mixup augmentation and switches to use L1 loss.
        """
        logger = get_root_logger()
        epoch = runner.epoch
        train_loader = runner.data_loader
        model = runner.model.module
        if epoch + 1 == runner.max_epochs - self.switch_epoch:
            logger.info('No mosaic and mixup aug now!')
            train_loader.dataset.enable_mosaic = False
            train_loader.dataset.enable_mixup = False
            logger.info('Add additional L1 loss now!')
            model.bbox_head.use_l1 = True
