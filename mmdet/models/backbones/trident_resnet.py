import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.backbones.resnet import Bottleneck, ResNet
from mmdet.models.builder import BACKBONES


class TridentConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            trident_dilations=(1, 2, 3),
            # num_branch=3,
            test_branch_idx=-1,
            bias=False,
            norm=None,
            activation=None,
    ):
        super(TridentConv, self).__init__()
        self.num_branch = len(trident_dilations)
        self.with_bias = bias
        self.test_branch_idx = test_branch_idx
        self.norm = norm
        self.activation = activation
        self.stride = stride
        self.kernel_size = kernel_size
        self.paddings = trident_dilations
        self.dilations = trident_dilations
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        # if isinstance(paddings, int):
        #     self.paddings = [paddings] * self.num_branch
        # if isinstance(trident_dilations, int):
        #     self.dilations = [trident_dilations] * self.num_branch

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_size,
                         self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def extra_repr(self):
        tmpstr = 'in_channels=' + str(self.in_channels)
        tmpstr += ', out_channels=' + str(self.out_channels)
        tmpstr += ', kernel_size=' + str(self.kernel_size)
        tmpstr += ', num_branch=' + str(self.num_branch)
        tmpstr += ', test_branch_idx=' + str(self.test_branch_idx)
        tmpstr += ', stride=' + str(self.stride)
        tmpstr += ', paddings=' + str(self.paddings)
        tmpstr += ', dilations=' + str(self.dilations)
        tmpstr += ', bias=' + str(self.bias)
        return tmpstr

    def forward(self, inputs):
        # num_branch = (self.num_branch
        #               if self.training or test_branch_idx == -1
        #               else 1)

        if self.training or self.test_branch_idx == -1:
            outputs = [
                F.conv2d(input, self.weight, self.bias, self.stride, padding,
                         dilation) for input, dilation, padding in zip(
                    inputs, self.dilations, self.paddings)
            ]
        else:
            assert len(inputs)==1
            outputs = [
                F.conv2d(inputs[0], self.weight, self.bias, self.stride,
                         self.paddings[self.test_branch_idx],
                         self.dilations[self.test_branch_idx])
            ]

        if self.norm is not None:
            outputs = [self.norm(x) for x in outputs]
        if self.activation is not None:
            outputs = [self.activation(x) for x in outputs]
        return outputs


class TridentBottleneckBlock(Bottleneck):

    def __init__(self,
                 trident_dilations=(1, 2, 3),
                 test_branch_idx=1,
                 concat_output=False,
                 **kwargs):

        super(TridentBottleneckBlock, self).__init__(**kwargs)
        self.trident_dilations = trident_dilations
        self.num_branch = len(trident_dilations)
        self.concat_output = concat_output
        self.test_branch_idx = test_branch_idx
        self.conv2 = TridentConv(
            self.planes,
            self.planes,
            kernel_size=3,
            stride=self.conv2_stride,
            # paddings=self.trident_dilations,
            bias=False,
            trident_dilations=self.trident_dilations,
            # num_branch=self.num_branch,
            test_branch_idx=test_branch_idx,
            norm=None)

    def forward(self, x):
        num_branch = (self.num_branch
                      if self.training or self.test_branch_idx == -1
                      else 1)
        identity = x
        if not isinstance(x, list):
            x = (x, ) * num_branch
            identity = x
            if self.downsample is not None:
                identity = [self.downsample(b) for b in x]

        out = [self.conv1(b) for b in x]
        out = [self.norm1(b) for b in out]
        out = [self.relu(b) for b in out]
        # if self.with_plugins:
        #     out = self.forward_plugin(out, self.after_conv1_plugin_names)

        out = self.conv2(out)
        out = [self.norm2(b) for b in out]
        out = [self.relu(b) for b in out]

        # if self.with_plugins:
        #     out = self.forward_plugin(out, self.after_conv2_plugin_names)

        out = [self.conv3(b) for b in out]
        out = [self.norm3(b) for b in out]

        # if self.with_plugins:
        #     out = self.forward_plugin(out, self.after_conv3_plugin_names)

        out = [out_b + identity_b for out_b, identity_b in zip(out, identity)]
        out = [self.relu(b) for b in out]
        if self.concat_output:
            out = torch.cat(out, dim=0)
        return out

    # def forward_plugin(self, x, plugin_names):
    #     out = x
    #     for (i, single_x) in enumerate(out):
    #         for name in plugin_names:
    #             out[i] = getattr(self, name)(single_x)
    #     return out


def make_trident_res_layer(block,
                           inplanes,
                           planes,
                           num_blocks,
                           stride=1,
                           trident_dilations=(1, 2, 3),
                           style='pytorch',
                           with_cp=False,
                           conv_cfg=None,
                           norm_cfg=dict(type='BN'),
                           dcn=None,
                           plugins=None,
                           test_branch_idx=-1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = []
        conv_stride = stride
        downsample.extend([
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=conv_stride,
                bias=False,
            ),
            build_norm_layer(norm_cfg, planes * block.expansion)[1]
        ])
        downsample = nn.Sequential(*downsample)

    layers = []
    for i in range(0, num_blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride if i == 0 else 1,
                trident_dilations=trident_dilations,
                downsample=downsample if i == 0 else None,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=plugins,
                test_branch_idx=test_branch_idx,
                concat_output=True if i == num_blocks - 1 else False))
        inplanes = planes * block.expansion
    return nn.Sequential(*layers)


@BACKBONES.register_module
class TridentResNet(ResNet):

    def __init__(self,
                 depth,
                 num_branch,
                 test_branch_idx,
                 trident_dilations=(1, 2, 3),
                 **kwargs):
        assert depth >= 50
        assert num_branch == len(trident_dilations)
        super(TridentResNet, self).__init__(depth, **kwargs)
        self.test_branch_idx = test_branch_idx
        self.num_branch = num_branch

        last_stage_idx = self.num_stages - 1
        stride = self.strides[last_stage_idx]
        dilation = trident_dilations
        dcn = self.dcn if self.stage_with_dcn[last_stage_idx] else None
        if self.plugins is not None:
            stage_plugins = self.make_stage_plugins(self.plugins,
                                                    last_stage_idx)
        else:
            stage_plugins = None
        planes = self.base_channels * 2 ** last_stage_idx
        res_layer = make_trident_res_layer(
            TridentBottleneckBlock,
            inplanes=self.block.expansion * self.base_channels *
                     2 ** (last_stage_idx - 1),
            planes=planes,
            num_blocks=self.stage_blocks[last_stage_idx],
            stride=stride,
            trident_dilations=dilation,
            style=self.style,
            with_cp=self.with_cp,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            dcn=dcn,
            plugins=stage_plugins,
            test_branch_idx=self.test_branch_idx
        )

        layer_name = f'layer{last_stage_idx + 1}'

        self.__setattr__(layer_name, res_layer)
        self.res_layers.pop(last_stage_idx)
        self.res_layers.insert(last_stage_idx, layer_name)

        self._freeze_stages()

        # self.feat_dim = self.block.expansion *
        #                 self.base_channels * 2 ** (
        #                         len(self.stage_blocks) - 1)

    # def forward(self, x):
    #     # for (k, v) in self.state_dict().items():
    #     #     if 'num_ba' in k:
    #     #         print("%035s %020s     %5d" %
    #     #               (k, str(list(v.shape)), v))
    #     #     else:
    #     #         print("%035s %020s     %5f     %.5f" %
    #     #               (k, str(list(v.shape)), v.mean().item(), v.std().item()))
    #     # asasas
    #     # import joblib
    #     # for (k, v) in self.named_parameters():
    #     #     nn.init.constant_(v, 0.001*(v.shape[0]//64)+0.001)
    #     #     print(k, 0.001*(v.shape[0]//64)+0.001)
    #     # x = joblib.load("/home/nirvana/temp/input.pkl").cuda()
    #
    #     if self.deep_stem:
    #         x = self.stem(x)
    #     else:
    #         x = self.conv1(x)
    #         x = self.norm1(x)
    #         x = self.relu(x)
    #     x = self.maxpool(x)
    #
    #     outs = []
    #     for i, layer_name in enumerate(self.res_layers):
    #         res_layer = getattr(self, layer_name)
    #         x = res_layer(x)
    #         if i in self.out_indices:
    #             outs.append(x)
    #     #         print(x.shape, x.mean().item(), x.std().item())
    #     #
    #     # asasasas
    #     return tuple(outs)


if __name__ == '__main__':
    net = TridentResNet(depth=50)
    # block = TridentBottleneckBlock(inplanes=128*4, planes=128)
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, TridentConv)):
            nn.init.constant_(m.weight, 0.05)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    import joblib

    # inputs = torch.randn([2, 128 * 4, 32, 32])
    # joblib.dump(inputs, "/home/nirvana/temp/inputs.pth")
    inputs = joblib.load('/home/nirvana/temp/inputs.pth')
    outputs = block(inputs)
    for k in range(3):
        print(outputs[k].mean().item(), outputs[k].std().item())
