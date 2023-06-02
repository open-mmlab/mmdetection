# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
from mmcv.cnn import VGG
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from ..necks import ssd_neck


@MODELS.register_module()
class SSDVGG(VGG, BaseModule):
    """VGG Backbone network for single-shot-detection.

    Args:
        depth (int): vgg 的深度, 可选深度有 {11, 13, 16, 19}.
        with_last_pool (bool): 是否在模型最后添加池化层
        ceil_mode (bool): 为True时,向上取整.为False时,向下取整.
        out_indices (Sequence[int]): 从哪个阶段输出.
        out_feature_indices (Sequence[int]): 从哪个特征图输出.
        init_cfg (dict or list[dict], optional): 初始化配置字典.

    Example:
        >>> self = SSDVGG(input_size=300, depth=11)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 300, 300)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 1024, 19, 19)
        (1, 512, 10, 10)
        (1, 256, 5, 5)
        (1, 256, 3, 3)
        (1, 256, 1, 1)
    """
    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }

    def __init__(self,
                 depth,
                 with_last_pool=False,
                 ceil_mode=True,
                 out_indices=(3, 4),
                 out_feature_indices=(22, 34),
                 init_cfg=None):
        # TODO: in_channels for mmcv.VGG
        super(SSDVGG, self).__init__(
            depth,
            with_last_pool=with_last_pool,
            ceil_mode=ceil_mode,
            out_indices=out_indices)
        # (conv + relu)*2+pool,*2+pool,*3+pool,*3+pool,*3
        # *3后没有pool是因为传入with_last_pool参数给删去了,下面是新增的层
        # MaxPool2d(3, 1, 1) 这里补上vgg中被删的第五个pool,但是stride=1
        # Conv2d(512, 1024, 3, padding=6, dilation=6) -> relu
        # Conv2d(1024, 1024, 1) -> relu
        self.features.add_module(
            str(len(self.features)),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.features.add_module(
            str(len(self.features)),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.features.add_module(
            str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.out_feature_indices = out_feature_indices

        if init_cfg is not None:
            self.init_cfg = init_cfg
        else:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(type='Constant', val=1, layer='BatchNorm2d'),
                dict(type='Normal', std=0.01, layer='Linear'),
            ]

    def init_weights(self, pretrained=None):
        # 初始化权重.由于继承顺序为VGG, BaseModule所以super(VGG, self)
        # 只会从VGG之后查找.即只会在BaseModule中查找init_weights方法
        super(VGG, self).init_weights()

    def forward(self, x):
        """Forward function."""
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_feature_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)
        # ssd300 outs (tensor(bs, 512, 38, 38), tensor(bs, 1024, 19, 19))


class L2Norm(ssd_neck.L2Norm):

    def __init__(self, **kwargs):
        super(L2Norm, self).__init__(**kwargs)
        warnings.warn('DeprecationWarning: L2Norm in ssd_vgg.py '
                      'is deprecated, please use L2Norm in '
                      'mmdet/models/necks/ssd_neck.py instead')
