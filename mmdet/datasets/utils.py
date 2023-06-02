# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

from mmcv.cnn import VGG
from mmcv.runner.hooks import HOOKS, Hook

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import (LoadAnnotations, LoadImageFromFile,
                                      LoadPanopticAnnotations)
from mmdet.models.dense_heads import GARPNHead, RPNHead
from mmdet.models.roi_heads.mask_heads import FusedSemanticHead


def replace_ImageToTensor(pipelines):
    """将数据管道中的 ImageToTensor 转换替换为 DefaultFormatBundle, 主要在批量推理场景下使用.

    Args:
        pipelines (list[dict]): 数据管道配置.

    Returns:
        list: 将所有 ImageToTensor 替换为 DefaultFormatBundle 的新数据管道配置.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='ImageToTensor', keys=['img']),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='DefaultFormatBundle'),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> assert expected_pipelines == replace_ImageToTensor(pipelines)
    """
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_ImageToTensor(
                pipeline['transforms'])
        elif pipeline['type'] == 'ImageToTensor':
            warnings.warn(
                '"ImageToTensor" 操作已被替换为 "DefaultFormatBundle",主要用于批量推理.'
                '不过还是建议在你的测试配置文件中手动替换它.', UserWarning)
            pipelines[i] = {'type': 'DefaultFormatBundle'}
    return pipelines


def get_loading_pipeline(pipeline):
    """Only keep loading image and annotations related configuration.

    Args:
        pipeline (list[dict]): Data pipeline configs.

    Returns:
        list[dict]: The new pipeline list with only keep
            loading image and annotations related configuration.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations', with_bbox=True),
        ...    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        ...    dict(type='RandomFlip', flip_ratio=0.5),
        ...    dict(type='Normalize', **img_norm_cfg),
        ...    dict(type='Pad', size_divisor=32),
        ...    dict(type='DefaultFormatBundle'),
        ...    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations', with_bbox=True)
        ...    ]
        >>> assert expected_pipelines ==\
        ...        get_loading_pipeline(pipelines)
    """
    loading_pipeline_cfg = []
    for cfg in pipeline:
        obj_cls = PIPELINES.get(cfg['type'])
        # TODO：use more elegant way to distinguish loading modules
        if obj_cls is not None and obj_cls in (LoadImageFromFile,
                                               LoadAnnotations,
                                               LoadPanopticAnnotations):
            loading_pipeline_cfg.append(cfg)
    assert len(loading_pipeline_cfg) == 2, \
        'The data pipeline in your config file must include ' \
        'loading image and annotations related pipeline.'
    return loading_pipeline_cfg


@HOOKS.register_module()
class NumClassCheckHook(Hook):

    def _check_head(self, runner):
        """Check whether the `num_classes` in head matches the length of
        `CLASSES` in `dataset`.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        model = runner.model
        dataset = runner.data_loader.dataset
        if dataset.CLASSES is None:
            runner.logger.warning(
                f'Please set `CLASSES` '
                f'in the {dataset.__class__.__name__} and'
                f'check if it is consistent with the `num_classes` '
                f'of head')
        else:
            assert type(dataset.CLASSES) is not str, \
                (f'`CLASSES` in {dataset.__class__.__name__}'
                 f'should be a tuple of str.'
                 f'Add comma if number of classes is 1 as '
                 f'CLASSES = ({dataset.CLASSES},)')
            for name, module in model.named_modules():
                if hasattr(module, 'num_classes') and not isinstance(
                        module, (RPNHead, VGG, FusedSemanticHead, GARPNHead)):
                    assert module.num_classes == len(dataset.CLASSES), \
                        (f'The `num_classes` ({module.num_classes}) in '
                         f'{module.__class__.__name__} of '
                         f'{model.__class__.__name__} does not matches '
                         f'the length of `CLASSES` '
                         f'{len(dataset.CLASSES)}) in '
                         f'{dataset.__class__.__name__}')

    def before_train_epoch(self, runner):
        """Check whether the training dataset is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._check_head(runner)

    def before_val_epoch(self, runner):
        """Check whether the dataset in val epoch is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._check_head(runner)
