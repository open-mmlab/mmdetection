from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class DataAugmentSwitchHook(Hook):
    """Trun off the mosaic data augment and switch off the loss, currently used in YOLOX.

    Args:
        no_aug_epoch (int): The epoch of close data augmentation. Default to 15.
    """
    def __init__(self, no_aug_epoch=15):
        self.no_aug_epoch = no_aug_epoch

    def before_train_epoch(self, runner):
        """close mosaic and mixup augmentation and additional L1 loss.
        """
        epoch = runner.epoch
        train_loader = runner.data_loader
        model = runner.model.module
        if epoch + 1 == runner.max_epochs - self.no_aug_epoch:
            print("--->No mosaic and mixup aug now!")
            train_loader.dataset.enable_mosaic = False
            train_loader.dataset.enable_mixup = False
            print("--->Add additional L1 loss now!")
            model.bbox_head.use_l1 = True
