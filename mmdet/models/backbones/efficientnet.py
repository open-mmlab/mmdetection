import torch.nn as nn

from torch.nn.modules.batchnorm import _BatchNorm
from ..registry import BACKBONES
import sys
sys.path.append('./mmdet/models/backbones')
import geffnet

@BACKBONES.register_module
class EfficientNet(nn.Module):
    """EfficientNet backbone and pretrained from https://github.com/rwightman/gen-efficientnet-pytorch
    Args:
        model_name (string): tf_efficientnet_b0-b7.
        pretrained (bool) : load pretrained weights, must be True.
        out_indices (Sequence[int]): Output from which stages. Should be (2, 3, 4, 5, 6) in EfficientDet.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Not used.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    Example:
        >>> from mmdet.models import EfficientNet
        >>> import torch
        >>> self = EfficientNet(model_name='tf_efficientnet_b2', pretrained=False)
        >>> self.eval()
        >>> inputs = torch.rand(1,3,768,768)
        >>> level_outputs = self(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 48, 96, 96)
        (1, 88, 48, 48)
        (1, 120, 24, 24)
        (1, 208, 12, 12)
        (1, 352, 6, 6)
    """

    def __init__(self,
                 model_name,
                 pretrained=True,
                 out_indices=(2, 3, 4, 5, 6),
                 style='pytorch',
                 frozen_stages=-1,
                 norm_eval=True):
        super(EfficientNet, self).__init__()
        self.out_indices = out_indices
        self.style = style
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.model = geffnet.create_model(model_name,pretrained=pretrained)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        return

    def forward(self, x):
        feature_map = self.model(x)
        outs=[]
        for i in self.out_indices:
            outs.append(feature_map[i])
        return tuple(outs)

    def train(self, mode=True): #need modify
        super(EfficientNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
