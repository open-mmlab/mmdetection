# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import VGG
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class NumClassCheckHook(Hook):
    """Check whether the `num_classes` in head matches the length of `classes`
    in `dataset.metainfo`."""

    def _check_head(self, runner: Runner, mode: str) -> None:
        """Check whether the `num_classes` in head matches the length of
        `classes` in `dataset.metainfo`.

        Args:
            runner (:obj:`Runner`): The runner of the training or evaluation
                process.
        """
        assert mode in ['train', 'val']
        model = runner.model
        dataset = runner.train_dataloader.dataset if mode == 'train' else \
            runner.val_dataloader.dataset
        if dataset.metainfo.get('classes', None) is None:
            runner.logger.warning(
                f'Please set `classes` '
                f'in the {dataset.__class__.__name__} `metainfo` and'
                f'check if it is consistent with the `num_classes` '
                f'of head')
        else:
            classes = dataset.metainfo['classes']
            assert type(classes) is not str, \
                (f'`classes` in {dataset.__class__.__name__}'
                 f'should be a tuple of str.'
                 f'Add comma if number of classes is 1 as '
                 f'classes = ({classes},)')
            from mmdet.models.roi_heads.mask_heads import FusedSemanticHead
            for name, module in model.named_modules():
                if hasattr(module, 'num_classes') and not name.endswith(
                        'rpn_head') and not isinstance(
                            module, (VGG, FusedSemanticHead)):
                    assert module.num_classes == len(classes), \
                        (f'The `num_classes` ({module.num_classes}) in '
                         f'{module.__class__.__name__} of '
                         f'{model.__class__.__name__} does not matches '
                         f'the length of `classes` '
                         f'{len(classes)}) in '
                         f'{dataset.__class__.__name__}')

    def before_train_epoch(self, runner: Runner) -> None:
        """Check whether the training dataset is compatible with head.

        Args:
            runner (:obj:`Runner`): The runner of the training or evaluation
                process.
        """
        self._check_head(runner, 'train')

    def before_val_epoch(self, runner: Runner) -> None:
        """Check whether the dataset in val epoch is compatible with head.

        Args:
            runner (:obj:`Runner`): The runner of the training or evaluation
                process.
        """
        self._check_head(runner, 'val')
