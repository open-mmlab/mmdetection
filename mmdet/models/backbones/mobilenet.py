import logging
from torch import nn
from ..registry import BACKBONES
from mmcv.cnn import (constant_init, kaiming_init, normal_init)
from mmcv.runner import load_checkpoint
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['MobileNetV2', 'mobilenet_v2']

model_urls = {
    'mobilenet_v2':
    'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim,
                       hidden_dim,
                       stride=stride,
                       groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@BACKBONES.register_module
class MobileNetV2(nn.Module):
    def __init__(self,
                 out_feature_indices=(
                     3,
                     6,
                     13,
                     18,
                 ),
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.out_feature_indices = out_feature_indices

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(
                inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(
                                 inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult,
                                        round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel,
                          output_channel,
                          stride,
                          expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(
            ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, num_classes),
        # )

    # def _forward(self, x):
    def forward(self, x):
        # x = self.features(x)
        # x = x.mean([2, 3])
        # x = self.classifier(x)
        # return x
        outs = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self.out_feature_indices:
                outs.append(x)
        return tuple(outs)

    # Allow for accessing forward method in a inherited class
    # forward = _forward
    # TODO: https://github.com/urbaneman/mmdetection_mobilenet/blob/master/mmdet/models/backbones/mobilenetv2.py
    def init_weights(self, pretrained=None):
        # weight initialization
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
            # state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=progress)
            # model.load_state_dict(state_dict)
            # state_dict = torch.load(pretrained)
            # if 'state_dict' in state_dict:
            #     state_dict = state_dict['state_dict']
            # if self.width_mult == 1.0:
            #     patched_dict = {}
            #     patched_dict = state_dict
            #     self.load_state_dict(patched_dict, strict=False)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    # if m.bias is not None:
                    #     nn.init.zeros_(m.bias)
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    # nn.init.ones_(m.weight)
                    # nn.init.zeros_(m.bias)
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    # nn.init.normal_(m.weight, 0, 0.01)
                    # nn.init.zeros_(m.bias)
                    normal_init(m, std=0.01)


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    import torch
    net = MobileNetV2()
    x = torch.randn(2, 3, 1024, 1024)
    y = net(x)
    # print(y)
