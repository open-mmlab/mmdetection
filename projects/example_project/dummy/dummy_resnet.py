# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import ResNet
from mmdet.registry import MODELS


@MODELS.register_module()
class DummyResNet(ResNet):
    """Implements a dummy ResNet wrapper for demonstration purpose.
    Args:
        **kwargs: All the arguments are passed to the parent class.
    """

    def __init__(self, **kwargs) -> None:
        print('Hello world!')
        super().__init__(**kwargs)
