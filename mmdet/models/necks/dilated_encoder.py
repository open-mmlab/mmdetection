import torch
import torch.nn as nn
from mmcv.cnn import caffe2_xavier_init
from torch.nn import BatchNorm2d, ReLU

from ..builder import NECKS


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels: int = 512,
                 mid_channels: int = 128,
                 dilation: int = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            BatchNorm2d(mid_channels), ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation), BatchNorm2d(mid_channels), ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            BatchNorm2d(in_channels), ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


@NECKS.register_module()
class DilatedEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, block_mid_channels,
                 num_residual_blocks):
        super(DilatedEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = [2, 4, 6, 8]
        self._init_layers()

    def _init_layers(self):
        self.lateral_conv = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1)
        self.lateral_norm = BatchNorm2d(self.out_channels)
        self.fpn_conv = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.fpn_norm = BatchNorm2d(self.out_channels)
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                Bottleneck(
                    self.out_channels,
                    self.block_mid_channels,
                    dilation=dilation))
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def init_weights(self):
        caffe2_xavier_init(self.lateral_conv)
        caffe2_xavier_init(self.fpn_conv)
        for m in [self.lateral_norm, self.fpn_norm]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature):
        out = self.lateral_norm(self.lateral_conv(feature[-1]))
        out = self.fpn_norm(self.fpn_conv(out))
        return self.dilated_encoder_blocks(out),
