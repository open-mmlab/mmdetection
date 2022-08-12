# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv

from ..builder import PIPELINES
from .compose import Compose


@PIPELINES.register_module()
class MultiScaleFlipAug:
    """包含多尺度和翻转的TTA.

    An example configuration is as followed:

    .. 该参数通常具有以下格式::

        img_scale=[(1333, 400), (1333, 800)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]

    使用上述配置的 MultiScaleFLipAug 之后,数据的内部元素将被包装到相同长度的列表中,如下所示:

    .. 经过TTA处理之后的data格式::

        dict(
            img=[...],
            img_shape=[...],
            scale=[(1333, 400), (1333, 400), (1333, 800), (1333, 800)]
            flip=[False, True, False, True]
            ...
        )

    Args:
        transforms (list[dict]): 在每个增强中应用的数据变换.
        img_scale (tuple | list[tuple] | None): 图像缩放的最大尺寸(各边).
        scale_factor (float | list[float] | None): 图像调整大小的各边比例因子.
        flip (bool): 是否应用翻转增强. Default: False.
        flip_direction (str | list[str]): 翻转方向,可选 "horizontal",
            "vertical" and "diagonal". 如果 flip_direction 是一个列表,
            将应用多个翻转增强.当 flip == False时将忽略该参数. Default:"horizontal".
    """

    def __init__(self,
                 transforms,
                 img_scale=None,
                 scale_factor=None,
                 flip=False,
                 flip_direction='horizontal'):
        self.transforms = Compose(transforms)
        assert (img_scale is None) ^ (scale_factor is None), (
            '二者必须且只能存在一个')
        if img_scale is not None:
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
            self.scale_key = 'scale'
            assert mmcv.is_list_of(self.img_scale, tuple)
        else:
            self.img_scale = scale_factor if isinstance(
                scale_factor, list) else [scale_factor]
            self.scale_key = 'scale_factor'

        self.flip = flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                '当 flip 设置为 False 时, 参数flip_direction会被忽略')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                '当 RandomFlip 不在transforms中时, flip参数无效')

    def __call__(self, results):
        """对results进行TTA操作.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
           dict[str: list]: The augmented data, 每个值都包含在一个列表中.
        """

        aug_data = []
        flip_args = [(False, None)]
        if self.flip:
            flip_args += [(True, direction)
                          for direction in self.flip_direction]
        for scale in self.img_scale:
            for flip, direction in flip_args:
                _results = results.copy()
                _results[self.scale_key] = scale
                _results['flip'] = flip
                _results['flip_direction'] = direction
                data = self.transforms(_results)
                aug_data.append(data)
        # list of dict to dict of list [{}]*n -> {[]*n} ,n代指TTA数量
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str
