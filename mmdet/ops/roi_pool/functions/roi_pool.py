import torch
from torch.autograd import Function

from .. import roi_pool_cuda


class RoIPoolFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError(
                '"out_size" must be an integer or tuple of integers')
        assert features.is_cuda
        ctx.save_for_backward(rois)
        num_channels = features.size(1)
        num_rois = rois.size(0)
        out_size = (num_rois, num_channels, out_h, out_w)
        output = features.new_zeros(out_size)
        argmax = features.new_zeros(out_size, dtype=torch.int)
        roi_pool_cuda.forward(features, rois, out_h, out_w, spatial_scale,
                              output, argmax)
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = features.size()
        ctx.argmax = argmax

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        spatial_scale = ctx.spatial_scale
        feature_size = ctx.feature_size
        argmax = ctx.argmax
        rois = ctx.saved_tensors[0]
        assert feature_size is not None

        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.new_zeros(feature_size)
            roi_pool_cuda.backward(grad_output.contiguous(), rois, argmax,
                                   spatial_scale, grad_input)

        return grad_input, grad_rois, None, None


roi_pool = RoIPoolFunction.apply
