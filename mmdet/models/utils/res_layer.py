from torch import nn as nn

from mmdet.models.utils import build_conv_layer, build_norm_layer


class ResLayer(nn.Sequential):

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 blocks,
                 stride=1,
                 dilation=1,
                 style='pytorch',
                 avg_down=False,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None,
                 gen_attention_blocks=[],
                 **block_kwargs):
        self.block = block
        self.inplanes = inplanes
        self.planes = planes
        self.blocks = blocks
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.avg_down = avg_down
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.gcb = gcb
        self.with_gcb = gcb is not None
        self.gen_attention = gen_attention
        self.with_gen_attention = gen_attention is not None

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention if
                (0 in gen_attention_blocks) else None,
                **block_kwargs))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation,
                    style=style,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    dcn=dcn,
                    gcb=gcb,
                    gen_attention=gen_attention if
                    (i in gen_attention_blocks) else None,
                    **block_kwargs))
        super(ResLayer, self).__init__(*layers)
