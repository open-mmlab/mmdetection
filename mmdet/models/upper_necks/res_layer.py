import logging

import torch.nn as nn

from ..backbones import ResNet, make_res_layer

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint


class ResLayer(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 depth,
                 layer_indicate=3,
                 stride=2,
                 dilation=1,
                 style='pytorch',
                 bn_eval=True,
                 bn_frozen=False,
                 with_cp=False):
        super(ResLayer, self).__init__()
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.layer_indicate = layer_indicate
        block, stage_blocks = ResNet.arch_settings[depth]
        stage_block = stage_blocks[layer_indicate]
        planes = 64 * 2**layer_indicate
        inplanes = 64 * 2**(layer_indicate - 1) * block.expansion

        res_layer = make_res_layer(
            block,
            inplanes,
            planes,
            stage_block,
            stride=stride,
            dilation=dilation,
            style=style,
            with_cp=with_cp)
        setattr(self, 'layer{}'.format(layer_indicate + 1), res_layer)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        res_layer = getattr(self, 'layer{}'.format(self.layer_indicate + 1))
        out = res_layer(x)
        return out

    def train(self, mode=True):
        super(ResLayer, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
