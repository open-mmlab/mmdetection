from ..registry import BACKBONES
from .resnet import ResNet


@BACKBONES.register_module
class ResNetV1d(ResNet):
    """ResNetV1d backbone.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)
