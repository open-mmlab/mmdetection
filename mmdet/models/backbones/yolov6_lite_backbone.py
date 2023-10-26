# Migrate based on Meituan yolov6 lite

import warnings
from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.layers.se_layer import SELayer

from mmdet.registry import MODELS
    

class Lite_EffiBlockS1(BaseModule):
    def __init__(self, 
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride):
        super(Lite_EffiBlockS1, self).__init__()
        self.conv_pw_1 = ConvModule(
            in_channels=in_channels // 2,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Hardswish')
        )
        self.conv_dw_1 = ConvModule(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels,
            norm_cfg=dict(type='BN')
        )
        self.se = SELayer(
            channels=mid_channels,
            ratio=4,
            act_cfg=(dict(type='ReLU'), dict(type='Hardsigmoid'))
        )
        self.conv_1 = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Hardswish')
        )
    
    def _channel_shuffle(x, groups):
        batchsize, num_channels, height, width = x.shape
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
    
    def forward(self, inputs):
        x1, x2 = torch.split(
            inputs,
            split_size_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            dim=1
        )
        x2 = self.conv_pw_1(x2)
        x3 = self.conv_dw_1(x2)
        x3 = self.se(x3)
        out = torch.cat([x1, x3], dim=1)
        return self._channel_shuffle(out, 2)

class Lite_EffiBlockS2(BaseModule):
    def __init__(self,
                       in_channels,
                       mid_channels,
                       out_channels,
                       stride):
        # branch 1
        self.conv_dw_1 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            norm_cfg=dict(type='BN')
        )
        self.conv_1 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Hardswish')
        )
        # branch 2
        self.conv_pw_2 = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Hardswish')
        )
        self.conv_dw_2 = ConvModule(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2,
            norm_cfg=dict(type='BN')
        )
        self.se = SELayer(
            channels=mid_channels // 2,
            ratio=4,
            act_cfg=(dict(type='ReLU'), dict(type='Hardsigmoid'))
        )
        self.conv_2 = ConvModule(
            in_channels=mid_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Hardswish')
        )
        self.conv_dw_3 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Hardswish')
        )
        self.conv_pw_3 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=11
        )
    
    def forward(self, inputs):
        # branch 1
        x1 = self.conv_dw_1(inputs)
        x1 = self.conv_1(x1)
        # branch 2
        x2 = self.conv_pw_2(inputs)
        x2 = self.conv_dw_2(x2)
        x2 = self.se(x2)
        x2 = self.conv_2(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self.conv_dw_3(out)
        out = self.conv_pw_3(out)
        return out
    

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
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Hardswish')
        )
        
        self.lite_effiblock1 = self._build_lite_effiblock(num_repeat[0],
                                                          out_channels[0],
                                                          mid_channels[1],
                                                          out_channels[1])
        
        self.lite_effiblock2 = self._build_lite_effiblock(num_repeat[1],
                                                          out_channels[1],
                                                          mid_channels[2],
                                                          out_channels[2])
        
        self.lite_effiblock3 = self._build_lite_effiblock(num_repeat[2],
                                                          out_channels[2],
                                                          mid_channels[3],
                                                          out_channels[3])
        
        self.lite_effiblock4 = self._build_lite_effiblock(num_repeat[3],
                                                          out_channels[3],
                                                          mid_channels[4],
                                                          out_channels[4])
    
    def forward(self, x):
        outputs = []
        x = self.conv0(x)
        x = self.lite_effiblock1(x)
        x = self.lite_effiblock2(x)
        outputs.append(x)
        x = self.lite_effiblock3(x)
        outputs.append(x)
        x = self.lite_effiblock4(x)
        outputs.append(x)
        return tuple(outputs)
    
    @staticmethod
    def _build_lite_effiblock(num_repeat, in_channels, mid_channels, out_channels):
        block_list = nn.Sequential()
        for i in range(num_repeat):
            if i == 0:
                block = Lite_EffiBlockS2(
                    in_channels=in_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    stride=2
                )
            else:
                block = Lite_EffiBlockS1(
                    in_channels=out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    stride=1
                )
            block_list.add_module(str(i), block)
        return block_list