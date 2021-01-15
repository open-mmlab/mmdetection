import torch.nn as nn
from ..builder import BACKBONES
from torchvision.models import shufflenetv2
import torch.nn.functional as F


@BACKBONES.register_module()
class ShuffleNet(nn.Module):
    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }

    def __init__(self, net_name='shufflenetv2_x1.0', stages_repeats=[4, 8, 4],
                 stages_out_channels=[24, 116, 232, 464, 1024], progress=True):
        super(ShuffleNet, self).__init__()
        self.model = shufflenetv2.ShuffleNetV2(stages_repeats, stages_out_channels)
        self.net_name = net_name
        self.progress = progress
        self.inplanes = 1024
        self.extra = self._make_extra_layers(self.extra_setting[300])

    def init_weights(self, pretrained=True):
        model_url = shufflenetv2.model_urls[self.net_name]
        state_dict = shufflenetv2.load_state_dict_from_url(model_url, progress=self.progress, check_hash=True)
        ret = self.model.load_state_dict(state_dict)
        print(ret)

    def forward(self, x):
        # append shufflenetv2_x1.0 output layers (19x19, 10x10)
        outputs = []
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        outputs.append(x)
        x = self.model.stage4(x)
        x = self.model.conv5(x)
        outputs.append(x)

        # append extra layers (5x5, 3x3, 2x2, 1x1)
        for i, layer in enumerate(self.extra):
            x = F.relu(layer(x), inplace=True)
            if i % 2 == 1:
                outputs.append(x)

        return tuple(outputs)

    def _make_extra_layers(self, outplanes):
        layers = []
        kernel_sizes = (1, 3)
        num_layers = 0
        outplane = None
        for i in range(len(outplanes)):
            if self.inplanes == 'S':
                self.inplanes = outplane
                continue
            k = kernel_sizes[num_layers % 2]
            if outplanes[i] == 'S':
                outplane = outplanes[i + 1]
                conv = nn.Conv2d(
                    self.inplanes, outplane, k, stride=2, padding=1)
            else:
                outplane = outplanes[i]
                conv = nn.Conv2d(
                    self.inplanes, outplane, k, stride=1, padding=0)
            layers.append(conv)
            self.inplanes = outplanes[i]
            num_layers += 1

        return nn.Sequential(*layers)

