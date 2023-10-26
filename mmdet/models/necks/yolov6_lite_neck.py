# Copied from Meituan yolov6 repo

import torch
from torch import nn
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.layers.csp_layer import CSPLayer


class Lite_EffiNeck(BaseModule):
    def __init__(
        self,
        in_channels,
        unified_channels
    ):
        super(Lite_EffiNeck, self).__init__()

        self.reduce_layer0 = ConvModule(
            in_channels=in_channels[0],
            out_channels=unified_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='HardSwish')
        )
        self.reduce_layer1 = ConvModule(
            in_channels=in_channels[1],
            out_channels=unified_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='HardSwish')
        )
        self.reduce_layer2 = ConvModule(
            in_channels=in_channels[2],
            out_channels=unified_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='HardSwish')
        )

        self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.csp_p4 = CSPLayer(
            in_channels=unified_channels * 2,
            out_channels=unified_channels,
            expand_ratio=0.5,
            num_blocks=1,
            add_identity=False,
            use_depthwise=True,
            channel_attention=False,
            use_cspnext_block=False,
            norm_cfg=dict(tyep='BN'),
            act_cfg=dict(type='HardSwish'),
            kernel_size=5
        )

        self.csp_p3 = CSPLayer(
            in_channels=unified_channels * 2,
            out_channels=unified_channels,
            expand_ratio=0.5,
            num_blocks=1,
            add_identity=False,
            use_depthwise=True,
            channel_attention=False,
            use_cspnext_block=False,
            norm_cfg=dict(tyep='BN'),
            act_cfg=dict(type='HardSwish'),
            kernel_size=5
        )

        self.csp_n4 = CSPLayer(
            in_channels=unified_channels * 2,
            out_channels=unified_channels,
            expand_ratio=0.5,
            num_blocks=1,
            add_identity=False,
            use_depthwise=True,
            channel_attention=False,
            use_cspnext_block=False,
            norm_cfg=dict(tyep='BN'),
            act_cfg=dict(type='HardSwish'),
            kernel_size=5
        )

        self.csp_n3 = CSPLayer(
            in_channels=unified_channels * 2,
            out_channels=unified_channels,
            expand_ratio=0.5,
            num_blocks=1,
            add_identity=False,
            use_depthwise=True,
            channel_attention=False,
            use_cspnext_block=False,
            norm_cfg=dict(tyep='BN'),
            act_cfg=dict(type='HardSwish'),
            kernel_size=5
        )

        self.downsample2 = DepthwiseSeparableConvModule(
            in_channels=unified_channels,
            out_channels=unified_channels,
            kernel_size=5,
            stride=2
        )

        self.downsample1 = DepthwiseSeparableConvModule(
            in_channels=unified_channels,
            out_channels=unified_channels,
            kernel_size=5,
            stride=2
        )

        self.p6_conv1 = DepthwiseSeparableConvModule(
            in_channels=unified_channels,
            out_channels=unified_channels,
            kernel_size=5,
            stride=2
        )

        self.p6_conv2 = DepthwiseSeparableConvModule(
            in_channels=unified_channels,
            out_channels=unified_channels,
            kernel_size=5,
            stride=2
        )
    
    def forward(self, inputs):
        # different level of features
        (x2, x1, x0) = inputs
        
        fpn_out0 = self.reduce_layer0(x0) #c5
        x1 = self.reduce_layer1(x1)       #c4
        x2 = self.reduce_layer2(x2)       #c3

        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out1 = self.csp_p4(f_concat_layer0)

        upsample_feat1 = self.upsample1(f_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out3 = self.csp_p3(f_concat_layer1) #p3

        down_feat1 = self.downsample2(pan_out3)
        p_concat_layer1 = torch.cat([down_feat1, f_out1], 1)
        pan_out2 = self.csp_n3(p_concat_layer1)  #p4

        down_feat0 = self.downsample1(pan_out2)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out1 = self.csp_n4(p_concat_layer2)  #p5

        top_features = self.p6_conv_1(fpn_out0)
        pan_out0 = top_features + self.p6_conv_2(pan_out1)  #p6

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]
        
        return outputs