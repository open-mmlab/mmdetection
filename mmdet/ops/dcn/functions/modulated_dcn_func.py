#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Function

from .. import modulated_dcn as _backend


class ModulatedDeformConvFunction(Function):

    def __init__(self, stride, padding, dilation=1, deformable_groups=1):
        super(ModulatedDeformConvFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

    def forward(self, input, offset, mask, weight, bias):
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or input.requires_grad:
            self.save_for_backward(input, offset, mask, weight, bias)
        output = input.new(*self._infer_shape(input, weight))
        self._bufs = [input.new(), input.new()]
        _backend.modulated_deform_conv_cuda_forward(
            input, weight, bias, self._bufs[0], offset, mask, output,
            self._bufs[1], weight.shape[2], weight.shape[3], self.stride,
            self.stride, self.padding, self.padding, self.dilation,
            self.dilation, self.deformable_groups)
        return output

    def backward(self, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = self.saved_tensors
        grad_input = input.new(*input.size()).zero_()
        grad_offset = offset.new(*offset.size()).zero_()
        grad_mask = mask.new(*mask.size()).zero_()
        grad_weight = weight.new(*weight.size()).zero_()
        grad_bias = bias.new(*bias.size()).zero_()
        _backend.modulated_deform_conv_cuda_backward(
            input, weight, bias, self._bufs[0], offset, mask, self._bufs[1],
            grad_input, grad_weight, grad_bias, grad_offset, grad_mask,
            grad_output, weight.shape[2], weight.shape[3], self.stride,
            self.stride, self.padding, self.padding, self.dilation,
            self.dilation, self.deformable_groups)

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias

    def _infer_shape(self, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * self.padding -
                      (self.dilation * (kernel_h - 1) + 1)) // self.stride + 1
        width_out = (width + 2 * self.padding -
                     (self.dilation * (kernel_w - 1) + 1)) // self.stride + 1
        return (n, channels_out, height_out, width_out)


class DeformRoIPoolingFunction(Function):

    def __init__(self,
                 spatial_scale,
                 pooled_size,
                 output_dim,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0):
        super(DeformRoIPoolingFunction, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

        assert self.trans_std >= 0.0 and self.trans_std <= 1.0

    def forward(self, data, rois, offset):
        if not data.is_cuda:
            raise NotImplementedError

        output = data.new(*self._infer_shape(data, rois))
        output_count = data.new(*self._infer_shape(data, rois))
        _backend.deform_psroi_pooling_cuda_forward(
            data, rois, offset, output, output_count, self.no_trans,
            self.spatial_scale, self.output_dim, self.group_size,
            self.pooled_size, self.part_size, self.sample_per_part,
            self.trans_std)

        # if data.requires_grad or rois.requires_grad or offset.requires_grad:
        #     self.save_for_backward(data, rois, offset, output_count)
        self.data = data
        self.rois = rois
        self.offset = offset
        self.output_count = output_count

        return output

    def backward(self, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError

        # data, rois, offset, output_count = self.saved_tensors
        data = self.data
        rois = self.rois
        offset = self.offset
        output_count = self.output_count
        grad_input = data.new(*data.size()).zero_()
        grad_offset = offset.new(*offset.size()).zero_()

        _backend.deform_psroi_pooling_cuda_backward(
            grad_output, data, rois, offset, output_count, grad_input,
            grad_offset, self.no_trans, self.spatial_scale, self.output_dim,
            self.group_size, self.pooled_size, self.part_size,
            self.sample_per_part, self.trans_std)
        return grad_input, torch.zeros(rois.shape).cuda(), grad_offset

    def _infer_shape(self, data, rois):
        # _, c, h, w = data.shape[:4]
        # c = data.shape[1]
        n = rois.shape[0]
        return n, self.output_dim, self.pooled_size, self.pooled_size
