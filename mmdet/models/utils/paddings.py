# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn.bricks.registry import PADDING_LAYERS

from .transformer import AdaptivePadding


@PADDING_LAYERS.register_module()
class samepadding(AdaptivePadding):

    def __init__(self, *args, **kwargs):
        kwargs['padding'] = 'same'
        super(samepadding, self).__init__(*args, **kwargs)
