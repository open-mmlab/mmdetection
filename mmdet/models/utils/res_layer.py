from torch import nn as nn

from mmdet.models.utils import build_conv_layer, build_norm_layer


class ResLayer(nn.Sequential):

    def __init__(self, block, inplanes, planes, num_blocks, **kwargs):
        self.block = block

        stride = kwargs.pop('stride')
        avg_down = kwargs.pop('avg_down')
        conv_cfg = kwargs.get('conv_cfg')
        norm_cfg = kwargs.get('norm_cfg')
        gen_attention = kwargs.pop('gen_attention')
        gen_attention_blocks = kwargs.pop('gen_attention_blocks')
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
                downsample=downsample,
                gen_attention=gen_attention
                if 0 in gen_attention_blocks else None,
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    gen_attention=gen_attention if
                    (i in gen_attention_blocks) else None,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)
