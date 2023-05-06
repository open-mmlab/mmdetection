# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet.registry import MODELS


@MODELS.register_module()
class GlobalAveragePooling(BaseModule):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.
    """

    def __init__(self, kernel_size=None, stride=None):
        super(GlobalAveragePooling, self).__init__()
        if kernel_size is None and stride is None:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AvgPool2d(kernel_size, stride)

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple([
                out.view(x.size(0),
                         torch.tensor(out.size()[1:]).prod())
                for out, x in zip(outs, inputs)
            ])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(
                inputs.size(0),
                torch.tensor(outs.size()[1:]).prod())
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
