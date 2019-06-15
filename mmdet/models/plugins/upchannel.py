import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init

from ..utils import ConvModule


class FPNUpChannels(nn.Module):
    """up channel module.
    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (dict): The config dict for convolution layers.
            (only applicable to conv_out)
        norm_cfg (dict): The config dict for normalization layers.
            (only applicable to conv_out)
    """
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None):

        super(FPNUpChannels, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        ## top
        self.top = ConvModule(
                     in_channels,
                     out_channels,
                     kernel_size=1,
                     conv_cfg=conv_cfg,
                     norm_cfg=norm_cfg,
                     activation=None)

        # self.top = nn.Sequential(
        #              nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        #              nn.BatchNorm2d(out_channels),
        #            )
        ## bottom
        self.bottom = nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(out_channels)
                      )
        #                    ConvModule(
        #                      in_channels,
        #                      in_channels,
        #                      kernel_size=3,
        #                      conv_cfg=conv_cfg,
        #                      norm_cfg=norm_cfg),
        #                    ConvModule(
        #                      in_channels,
        #                      out_channels,
        #                      kernel_size=1,
        #                      conv_cfg=conv_cfg,
        #                      norm_cfg=norm_cfg, 
        #                      activation=None),

        # Conv2d(num_inputs, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        # nn.BatchNorm2d(out_channels)

    def forward(self, x):
        ## top
        out = self.top(x)
        out0 = self.bottom(x)
        ## bottom
        ## residual
        out1 = out + out0
      
        out1 = self.relu(out1)
        return out1



class FPNFFConv(nn.Module):
    def __init__(self, in_channels):
        super(FPNFFConv, self).__init__()

        inter_channels = in_channels // 4
        out_channels = in_channels

        self.relu = nn.ReLU(inplace=True)
        ## top
        self.bottleneck = nn.Sequential(
                             nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.BatchNorm2d(inter_channels),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
                             nn.BatchNorm2d(inter_channels),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.BatchNorm2d(out_channels)
          )

        # Conv2d(num_inputs, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        # nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        ## bottom
        out = self.bottleneck(x)
        ## residual
        out1 = out + identity
        out1 = self.relu(out1)

        return out1


