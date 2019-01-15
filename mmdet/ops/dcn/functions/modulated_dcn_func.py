#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Function

from .. import modulated_dcn_cuda as _backend


class ModulatedDeformConvFunction(Function):

    def __init__(ctx, stride, padding, dilation=1, deformable_groups=1):
        super(ModulatedDeformConvFunction, ctx).__init__()
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.deformable_groups = deformable_groups

    @staticmethod
    def forward(ctx,
                input,
                offset,
                mask,
                weight,
                bias,
                stride,
                padding,
                dilation=1,
                deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.deformable_groups = deformable_groups
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new(
            *ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new(), input.new()]
        _backend.modulated_deform_conv_cuda_forward(
            input, weight, bias, ctx._bufs[0], offset, mask, output,
            ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride,
            ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation,
            ctx.deformable_groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        _backend.modulated_deform_conv_cuda_backward(
            input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1],
            grad_input, grad_weight, grad_bias, grad_offset, grad_mask,
            grad_output, weight.shape[2], weight.shape[3], ctx.stride,
            ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation,
            ctx.deformable_groups)

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding -
                      (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding -
                     (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx,
                data,
                rois,
                offset,
                spatial_scale,
                pooled_size,
                output_dim,
                no_trans,
                group_size=1,
                part_size=None,
                sample_per_part=4,
                trans_std=.0):
        ctx.spatial_scale = spatial_scale
        ctx.pooled_size = pooled_size
        ctx.output_dim = output_dim
        ctx.no_trans = no_trans
        ctx.group_size = group_size
        ctx.part_size = pooled_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std

        assert 0.0 <= ctx.trans_std <= 1.0
        if not data.is_cuda:
            raise NotImplementedError

        output = data.new(
            *DeformRoIPoolingFunction._infer_shape(ctx, data, rois))
        output_count = data.new(
            *DeformRoIPoolingFunction._infer_shape(ctx, data, rois))
        _backend.deform_psroi_pooling_cuda_forward(
            data, rois, offset, output, output_count, ctx.no_trans,
            ctx.spatial_scale, ctx.output_dim, ctx.group_size, ctx.pooled_size,
            ctx.part_size, ctx.sample_per_part, ctx.trans_std)

        # if data.requires_grad or rois.requires_grad or offset.requires_grad:
        #     ctx.save_for_backward(data, rois, offset, output_count)
        ctx.data = data
        ctx.rois = rois
        ctx.offset = offset
        ctx.output_count = output_count

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError

        # data, rois, offset, output_count = ctx.saved_tensors
        data = ctx.data
        rois = ctx.rois
        offset = ctx.offset
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_offset = torch.zeros_like(offset)

        _backend.deform_psroi_pooling_cuda_backward(
            grad_output, data, rois, offset, output_count, grad_input,
            grad_offset, ctx.no_trans, ctx.spatial_scale, ctx.output_dim,
            ctx.group_size, ctx.pooled_size, ctx.part_size,
            ctx.sample_per_part, ctx.trans_std)
        return (grad_input, torch.zeros_like(rois), grad_offset, None, None,
                None, None, None, None, None, None)

    @staticmethod
    def _infer_shape(ctx, data, rois):
        n = rois.shape[0]
        return n, ctx.output_dim, ctx.pooled_size, ctx.pooled_size


modulated_deform_conv = ModulatedDeformConvFunction.apply
deform_roi_pooling = DeformRoIPoolingFunction.apply
