import math

import torch.nn as nn
from mmcv.runner import load_checkpoint

vgg_config = {
    'SSD300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512,
               512, 512, 'M', 512, 512, 512],
    'SSD512': [],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
              512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
              'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
              512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VggNet(nn.Module):
    def __init__(self, vggtype, out_indices=(22, 34)):
        super(VggNet, self).__init__()
        self.features = nn.ModuleList(make_layers(vggtype))
        if vggtype == 'SSD300':
            pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
            conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
            self.features += [pool5, conv6, nn.ReLU(inplace=True), conv7,
                              nn.ReLU(inplace=True)]
        self.out_indices = out_indices

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
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)


def make_layers(vggtype='SSD300', batch_norm=False):
    cfg = vgg_config[vggtype]
    layers = []
    in_channels = 3
    for out_channel in cfg:
        if out_channel == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif out_channel == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, out_channel, kernel_size=3,
                               padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channel),
                           nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = out_channel
    return layers
