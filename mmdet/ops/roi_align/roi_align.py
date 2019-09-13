import torch.nn as nn
import torch.onnx.symbolic_helper as sym_help
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torch.onnx.symbolic_opset9 import reshape
from torch.onnx.symbolic_opset10 import _slice

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
            if rois.numel() > 0:
                roi_align_cuda.forward(features, rois, out_h, out_w,
                                       spatial_scale, sample_num, output)
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

        return grad_input, grad_rois, None, None, None, None

    @staticmethod
    def symbolic(g, features, rois, out_size, spatial_scale, sample_num=0):
        batch_indices = reshape(
            g,
            g.op(
                'Cast',
                _slice(g, rois, axes=[1], starts=[0], ends=[1]),
                to_i=sym_help.cast_pytorch_to_onnx['Long']), [-1])
        bboxes = _slice(g, rois, axes=[1], starts=[1], ends=[5])
        out_h, out_w = _pair(out_size)
        return g.op(
            'RoiAlign',
            features,
            bboxes,
            batch_indices,
            output_height_i=out_h,
            output_width_i=out_w,
            sampling_ratio_i=sample_num,
            spatial_scale_f=spatial_scale)


roi_align = RoIAlignFunction.apply


class RoIAlign(nn.Module):

    def __init__(self,
                 out_size,
                 spatial_scale,
                 sample_num=0,
                 use_torchvision=False):
        super(RoIAlign, self).__init__()

        self.out_size = _pair(out_size)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.use_torchvision = use_torchvision

    def forward(self, features, rois):
        if self.use_torchvision:
            from torchvision.ops import roi_align as tv_roi_align
            return tv_roi_align(features, rois, self.out_size,
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
