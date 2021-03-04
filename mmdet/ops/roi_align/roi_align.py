import torch
import torch.nn as nn
import torch.onnx.symbolic_helper as sym_help
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torch.onnx.symbolic_opset9 import reshape, sub
from torch.onnx.symbolic_opset10 import _slice

from . import roi_align_ext


class RoIAlignFunction(Function):

    @staticmethod
    def forward(ctx,
                features,
                rois,
                out_size,
                spatial_scale,
                sample_num=0,
                aligned=True):
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()
        ctx.aligned = aligned

        if aligned:
            output = roi_align_ext.forward_v2(features, rois, spatial_scale,
                                              out_h, out_w, sample_num,
                                              aligned)
        elif features.is_cuda:
            (batch_size, num_channels, data_height,
             data_width) = features.size()
            num_rois = rois.size(0)

            output = features.new_zeros(num_rois, num_channels, out_h, out_w)
            roi_align_ext.forward_v1(features, rois, out_h, out_w,
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
        aligned = ctx.aligned
        assert feature_size is not None

        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output.size(3)
        out_h = grad_output.size(2)

        grad_input = grad_rois = None
        if not aligned:
            if ctx.needs_input_grad[0]:
                grad_input = rois.new_zeros(batch_size, num_channels,
                                            data_height, data_width)
                roi_align_ext.backward_v1(grad_output.contiguous(), rois,
                                          out_h, out_w, spatial_scale,
                                          sample_num, grad_input)
        else:
            grad_input = roi_align_ext.backward_v2(grad_output, rois,
                                                   spatial_scale, out_h, out_w,
                                                   batch_size, num_channels,
                                                   data_height, data_width,
                                                   sample_num, aligned)

        return grad_input, grad_rois, None, None, None, None

    @staticmethod
    def symbolic(g, features, rois, out_size, spatial_scale, sample_num=0, aligned=True):
        batch_indices = reshape(
            g,
            g.op(
                'Cast',
                _slice(g, rois, axes=[1], starts=[0], ends=[1]),
                to_i=sym_help.cast_pytorch_to_onnx['Long']), [-1])
        bboxes = _slice(g, rois, axes=[1], starts=[1], ends=[5])
        if aligned:
            scale = sym_help._maybe_get_scalar(spatial_scale)
            offset = g.op("Constant", value_t=torch.tensor(0.5 / scale, dtype=torch.float32))
            bboxes = sub(g, bboxes, offset)
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
                 output_size,
                 spatial_scale,
                 sampling_ratio=0,
                 use_torchvision=False,
                 aligned=True):
        """
        Args:
            out_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sample_num (int): number of inputs samples to take for each
                output sample. 2 to take samples densely for current models.
            use_torchvision (bool): whether to use roi_align from torchvision
            aligned (bool): if False, use the legacy implementation in
                MMDetection. If True, align the results more perfectly.

        Note:
            The implementation of RoIAlign when aligned=True is modified from
            https://github.com/facebookresearch/detectron2/

            The meaning of aligned=True:

            Given a continuous coordinate c, its two neighboring pixel
            indices (in our pixel model) are computed by floor(c - 0.5) and
            ceil(c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal
            at continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing
            neighboring pixel indices and therefore it uses pixels with a
            slightly incorrect alignment (relative to our pixel model) when
            performing bilinear interpolation.

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors;

            The difference does not make a difference to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(RoIAlign, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.aligned = aligned
        self.sampling_ratio = int(sampling_ratio)
        self.use_torchvision = use_torchvision
        assert not (use_torchvision and
                    aligned), 'Torchvision does not support aligned RoIAlgin'

    def forward(self, features, rois):
        """
        Args:
            features: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4
            columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5

        if self.use_torchvision:
            from torchvision.ops import roi_align as tv_roi_align
            return tv_roi_align(features, rois, self.output_size,
                                self.spatial_scale, self.sampling_ratio)
        else:
            return roi_align(features, rois, self.output_size, self.spatial_scale,
                             self.sampling_ratio, self.aligned)

    def __repr__(self):
        indent_str = '\n    '
        format_str = self.__class__.__name__
        format_str += f'({indent_str}out_size={self.output_size},'
        format_str += f'{indent_str}spatial_scale={self.spatial_scale},'
        format_str += f'{indent_str}sampling_ratio={self.sampling_ratio},'
        format_str += f'{indent_str}use_torchvision={self.use_torchvision},'
        format_str += f'{indent_str}aligned={self.aligned})'
        return format_str
