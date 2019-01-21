import torch
from torch.autograd import Function

from .. import deform_pool_cuda


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

        output = data.new_empty(
            DeformRoIPoolingFunction._infer_shape(ctx, data, rois))
        output_count = data.new_empty(
            DeformRoIPoolingFunction._infer_shape(ctx, data, rois))
        deform_pool_cuda.deform_psroi_pooling_cuda_forward(
            data, rois, offset, output, output_count, ctx.no_trans,
            ctx.spatial_scale, ctx.output_dim, ctx.group_size, ctx.pooled_size,
            ctx.part_size, ctx.sample_per_part, ctx.trans_std)

        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, rois, offset)
        # ctx.data = data
        # ctx.rois = rois
        # ctx.offset = offset
        ctx.output_count = output_count

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError

        data, rois, offset = ctx.saved_tensors
        # data = ctx.data
        # rois = ctx.rois
        # offset = ctx.offset
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_offset = torch.zeros_like(offset)

        deform_pool_cuda.deform_psroi_pooling_cuda_backward(
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


deform_roi_pooling = DeformRoIPoolingFunction.apply
