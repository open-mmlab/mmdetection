import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from . import roi_pool_cuda


class RoIPoolFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale):
        assert features.is_cuda
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
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
    @once_differentiable
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


class RoIPool(nn.Module):

    def __init__(self, out_size, spatial_scale, use_torchvision=False):
        super(RoIPool, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.use_torchvision = use_torchvision

    def forward(self, features, rois):
        if self.use_torchvision:
            from torchvision.ops import roi_pool as tv_roi_pool
            return tv_roi_pool(features, rois, _pair(self.out_size),
                               self.spatial_scale)
        else:
            return roi_pool(features, rois, self.out_size, self.spatial_scale)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}'.format(
            self.out_size, self.spatial_scale)
        format_str += ', use_torchvision={})'.format(self.use_torchvision)
        return format_str
