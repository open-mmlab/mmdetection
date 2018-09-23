import math
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.torchpack import load_checkpoint


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='fb'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='fb',
                 with_cp=False):
        """Bottleneck block
        if style is "fb", the stride-two layer is the 3x3 conv layer,
        if style is "msra", the stride-two layer is the first 1x1 conv layer
        """
        super(Bottleneck, self).__init__()
        assert style in ['fb', 'msra']
        if style == 'fb':
            conv1_stride = 1
            conv2_stride = stride
        else:
            conv1_stride = stride
            conv2_stride = 1
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=conv1_stride, bias=False)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):

        def _inner_forward(x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='fb',
                   with_cp=False):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            with_cp=with_cp))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation, style=style, with_cp=with_cp))

    return nn.Sequential(*layers)


class ResHead(nn.Module):

    def __init__(self, block, num_blocks, stride=2, dilation=1, style='fb'):
        self.layer4 = make_res_layer(
            block,
            1024,
            512,
            num_blocks,
            stride=stride,
            dilation=dilation,
            style=style)

    def forward(self, x):
        return self.layer4(x)


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 style='fb',
                 sync_bn=False,
                 with_cp=False):
        super(ResNet, self).__init__()
        if not len(layers) == len(strides) == len(dilations):
            raise ValueError(
                'The number of layers, strides and dilations must be equal, '
                'but found have {} layers, {} strides and {} dilations'.format(
                    len(layers), len(strides), len(dilations)))
        assert max(out_indices) < len(layers)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.style = style
        self.sync_bn = sync_bn
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_layers = []
        for i, num_blocks in enumerate(layers):

            stride = strides[i]
            dilation = dilations[i]

            layer_name = 'layer{}'.format(i + 1)
            planes = 64 * 2**i
            res_layer = make_res_layer(
                block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp)
            self.inplanes = planes * block.expansion
            setattr(self, layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.feat_dim = block.expansion * 64 * 2**(len(layers) - 1)
        self.with_cp = with_cp

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        if not self.sync_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False


resnet_cfg = {
    18: (BasicBlock, (2, 2, 2, 2)),
    34: (BasicBlock, (3, 4, 6, 3)),
    50: (Bottleneck, (3, 4, 6, 3)),
    101: (Bottleneck, (3, 4, 23, 3)),
    152: (Bottleneck, (3, 8, 36, 3))
}


def resnet(depth,
           num_stages=4,
           strides=(1, 2, 2, 2),
           dilations=(1, 1, 1, 1),
           out_indices=(2, ),
           frozen_stages=-1,
           style='fb',
           sync_bn=False,
           with_cp=False):
    """Constructs a ResNet model.

    Args:
        depth (int): depth of resnet, from {18, 34, 50, 101, 152}
        num_stages (int): num of resnet stages, normally 4
        strides (list): strides of the first block of each stage
        dilations (list): dilation of each stage
        out_indices (list): output from which stages
    """
    if depth not in resnet_cfg:
        raise KeyError('invalid depth {} for resnet'.format(depth))
    block, layers = resnet_cfg[depth]
    model = ResNet(block, layers[:num_stages], strides, dilations, out_indices,
                   frozen_stages, style, sync_bn, with_cp)
    return model
