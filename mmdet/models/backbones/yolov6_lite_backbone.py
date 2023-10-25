# Migrate based on Meituan yolov6 lite

import warnings
from typing import Sequence, Tuple, Union
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from mmdet.registry import MODELS
    

@MODELS.register_module()
class Lite_EffiBackbone(BaseModule):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: list,
                 num_repeat: Sequence[int] = [1, 3, 7, 3],
                 ) -> None:
        super(Lite_EffiBackbone, self).__init__()
        # TODO: why?
        out_channels[0] = 24
        self.conv0 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels[0],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='HardSwish')
        )
    
    
    def _build_effilite_block(num_repeat, in_channels, mid_channels, out_channels):
        
        

class Lite_EffiBackbone(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 num_repeat=[1, 3, 7, 3]
                 ):
        super().__init__()
        out_channels[0] = 24
        self.conv_0 = ConvBNHS(in_channels=in_channels,
                               out_channels=out_channels[0],
                               kernel_size=3,
                               stride=2,
                               padding=1)

        self.lite_effiblock_1 = self.build_block(num_repeat[0],
                                                 out_channels[0],
                                                 mid_channels[1],
                                                 out_channels[1])

        self.lite_effiblock_2 = self.build_block(num_repeat[1],
                                                 out_channels[1],
                                                 mid_channels[2],
                                                 out_channels[2])

        self.lite_effiblock_3 = self.build_block(num_repeat[2],
                                                 out_channels[2],
                                                 mid_channels[3],
                                                 out_channels[3])

        self.lite_effiblock_4 = self.build_block(num_repeat[3],
                                                 out_channels[3],
                                                 mid_channels[4],
                                                 out_channels[4])

    def forward(self, x):
        outputs = []
        x = self.conv_0(x)
        x = self.lite_effiblock_1(x)
        x = self.lite_effiblock_2(x)
        outputs.append(x)
        x = self.lite_effiblock_3(x)
        outputs.append(x)
        x = self.lite_effiblock_4(x)
        outputs.append(x)
        return tuple(outputs)

    @staticmethod
    def build_block(num_repeat, in_channels, mid_channels, out_channels):
        block_list = nn.Sequential()
        for i in range(num_repeat):
            if i == 0:
                block = Lite_EffiBlockS2(
                    in_channels=in_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    stride=2)
            else:
                block = Lite_EffiBlockS1(
                    in_channels=out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    stride=1)
            block_list.add_module(str(i), block)
        return block_list
