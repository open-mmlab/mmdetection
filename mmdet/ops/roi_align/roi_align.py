import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from . import roi_align_cuda


class RoIAlignFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale, sample_num=0):
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new_zeros(num_rois, num_channels, out_h, out_w)
        if features.is_cuda:
            roi_align_cuda.forward(features, rois, out_h, out_w, spatial_scale,
                                   sample_num, output)
        else:
            raise NotImplementedError

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        assert (feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output.size(3)
        out_h = grad_output.size(2)

        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, data_height,
                                        data_width)
            roi_align_cuda.backward(grad_output.contiguous(), rois, out_h,
                                    out_w, spatial_scale, sample_num,
                                    grad_input)

        return grad_input, grad_rois, None, None, None


roi_align = RoIAlignFunction.apply


class RoIAlign(nn.Module):

    def __init__(self,
                 out_size,
                 spatial_scale,
                 sample_num=0,
                 use_torchvision=False):
        super(RoIAlign, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.use_torchvision = use_torchvision

    def forward(self, features, rois):
        if self.use_torchvision:
            from torchvision.ops import roi_align as tv_roi_align
            return tv_roi_align(features, rois, _pair(self.out_size),
                                self.spatial_scale, self.sample_num)
        else:
            return roi_align(features, rois, self.out_size, self.spatial_scale,
                             self.sample_num)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}, sample_num={}'.format(
            self.out_size, self.spatial_scale, self.sample_num)
        format_str += ', use_torchvision={})'.format(self.use_torchvision)
        return format_str
