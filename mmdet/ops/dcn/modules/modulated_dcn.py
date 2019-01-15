#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
from torch import nn
from torch.nn.modules.utils import _pair

from ..functions.modulated_dcn_func import DeformRoIPoolingFunction
from ..functions.modulated_dcn_func import ModulatedDeformConvFunction


class ModulatedDeformConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 deformable_groups=1,
                 no_bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.no_bias = no_bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()
        if self.no_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        func = ModulatedDeformConvFunction(
            self.stride, self.padding, self.dilation, self.deformable_groups)
        return func(input, offset, mask, self.weight, self.bias)


class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 deformable_groups=1,
                 no_bias=False):
        super(ModulatedDeformConvPack,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, deformable_groups, no_bias)

        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        func = ModulatedDeformConvFunction(
            self.stride, self.padding, self.dilation, self.deformable_groups)
        return func(input, offset, mask, self.weight, self.bias)


class DeformRoIPooling(nn.Module):

    def __init__(self,
                 spatial_scale,
                 pooled_size,
                 output_dim,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.out_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std
        self.func = DeformRoIPoolingFunction(
            self.spatial_scale, self.pooled_size, self.output_dim,
            self.no_trans, self.group_size, self.part_size,
            self.sample_per_part, self.trans_std)

    def forward(self, data, rois, offset):

        if self.no_trans:
            offset = data.new()
        return self.func(data, rois, offset)


class ModulatedDeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self,
                 spatial_scale,
                 pooled_size,
                 output_dim,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0,
                 deform_fc_dim=1024):
        super(ModulatedDeformRoIPoolingPack, self).__init__(
            spatial_scale, pooled_size, output_dim, no_trans, group_size,
            part_size, sample_per_part, trans_std)

        self.deform_fc_dim = deform_fc_dim

        if not no_trans:
            self.func_offset = DeformRoIPoolingFunction(
                self.spatial_scale, self.pooled_size, self.output_dim, True,
                self.group_size, self.part_size, self.sample_per_part,
                self.trans_std)
            self.offset_fc = nn.Sequential(
                nn.Linear(
                    self.pooled_size * self.pooled_size * self.output_dim,
                    self.deform_fc_dim), nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim,
                          self.pooled_size * self.pooled_size * 2))
            self.offset_fc[4].weight.data.zero_()
            self.offset_fc[4].bias.data.zero_()
            self.mask_fc = nn.Sequential(
                nn.Linear(
                    self.pooled_size * self.pooled_size * self.output_dim,
                    self.deform_fc_dim), nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim,
                          self.pooled_size * self.pooled_size * 1),
                nn.Sigmoid())
            self.mask_fc[2].weight.data.zero_()
            self.mask_fc[2].bias.data.zero_()

    def forward(self, data, rois):
        if self.no_trans:
            offset = data.new()
        else:
            n = rois.shape[0]
            offset = data.new()
            x = self.func_offset(data, rois, offset)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.pooled_size, self.pooled_size)
            mask = self.mask_fc(x.view(n, -1))
            mask = mask.view(n, 1, self.pooled_size, self.pooled_size)
            feat = self.func(data, rois, offset) * mask
            return feat
        return self.func(data, rois, offset)
