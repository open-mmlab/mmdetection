import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init

from ..utils import ConvModule


class NonLocal2D(nn.Module):

    def __init__(self,
                 in_channels,
                 reduction=2,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian'):
        super(NonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian']

        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.theta = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.phi = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=None)

        self.init_weights()

    def init_weights(self):
        constant_init(self.conv_out.conv, 0)

    def embedded_gaussian(self, x):
        # x: [N, C, H, W]
        n = x.size(0)

        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        # g_x: [N, HxW, C]
        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # theta_x: [N, HxW, C]
        phi_x = self.phi(x).view(n, self.inter_channels, -1)
        # phi_x: [N, C, HxW]

        map_t_p = torch.matmul(theta_x, phi_x)
        # map_t_p: [N, HxW, HxW]
        mask_t_p = F.softmax(map_t_p, dim=-1)

        map_ = torch.matmul(mask_t_p, g_x)
        map_ = map_.reshape(n, self.inter_channels, x.size(2), x.size(3))
        mask = self.conv_out(map_)
        final = mask + x
        return final

    def forward(self, x):
        if self.mode == 'embedded_gaussian':
            output = self.embedded_gaussian(x)
        else:
            raise NotImplementedError('The code has not been implemented.')
        return output
