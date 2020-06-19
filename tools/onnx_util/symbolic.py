"""Modified from https://github.com/pytorch/pytorch."""
import onnx_util.symbolic_helper as sym_help
import torch
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_registry import register_op


def _interpolate(name, dim, interpolate_mode):

    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = sym_help._get_interpolate_attributes(
            g, interpolate_mode, args)
        align_corners = sym_help._maybe_get_scalar(align_corners)
        transformation_mode = 'asymmetric' \
            if interpolate_mode == 'nearest' \
            else 'align_corners' if align_corners else 'pytorch_half_pixel'
        empty_tensor = g.op(
            'Constant', value_t=torch.tensor([], dtype=torch.float32))

        if scales is None:
            input_size = g.op('Shape', input)
            input_size_beg = sym_help._slice_helper(
                g, input_size, axes=[0], ends=[2], starts=[0])
            output_size = g.op(
                'Cast',
                output_size,
                to_i=sym_help.cast_pytorch_to_onnx['Long'])
            output_size = g.op('Concat', input_size_beg, output_size, axis_i=0)
            scales = g.op(
                'Constant', value_t=torch.tensor([], dtype=torch.float32))
            return g.op(
                'Resize',
                input,
                empty_tensor,
                # roi only takes effect whith
                # coordinate_transformation_mode="tf_crop_and_resize"
                scales,  # scales is not needed since we are sending out_size
                output_size,
                coordinate_transformation_mode_s=transformation_mode,
                cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                mode_s=interpolate_mode,  # nearest, linear, or cubic
                nearest_mode_s='floor')  # only valid when mode="nearest"
        else:
            return g.op(
                'Resize',
                input,
                empty_tensor,
                # roi only takes effect with
                # coordinate_transformation_mode="tf_crop_and_resize"
                scales,  # scales is not needed since we are sending out_size
                coordinate_transformation_mode_s=transformation_mode,
                cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                mode_s=interpolate_mode,  # nearest, linear, or cubic
                nearest_mode_s='floor')  # only valid when mode="nearest"

    return symbolic_fn


upsample_nearest1d = _interpolate('upsample_nearest1d', 3, 'nearest')
upsample_nearest2d = _interpolate('upsample_nearest2d', 4, 'nearest')
upsample_nearest3d = _interpolate('upsample_nearest3d', 5, 'nearest')
upsample_linear1d = _interpolate('upsample_linear1d', 3, 'linear')
upsample_bilinear2d = _interpolate('upsample_bilinear2d', 4, 'linear')
upsample_trilinear3d = _interpolate('upsample_trilinear3d', 5, 'linear')
upsample_bicubic2d = _interpolate('upsample_bicubic2d', 4, 'cubic')


@parse_args('v', 'v', 'i', 'i', 'i', 'none')
def topk(g, self, k, dim, largest, sorted, out=None):
    return sym_help._topk_helper(
        g, self, k, dim, largest=largest, sorted=sorted, out=out)


def masked_select(g, self, mask):
    from torch.onnx.symbolic_opset9 import nonzero, expand_as
    index = nonzero(g, expand_as(g, mask, self))
    return g.op('GatherND', self, index)


def register_extra_symbolics(opset=11):
    register_op('topk', topk, '', opset)
    register_op('masked_select', masked_select, '', opset)
    register_op('upsample_nearest1d', upsample_nearest1d, '', opset)
    register_op('upsample_nearest2d', upsample_nearest2d, '', opset)
    register_op('upsample_nearest3d', upsample_nearest3d, '', opset)
    register_op('upsample_linear1d', upsample_linear1d, '', opset)
    register_op('upsample_bilinear2d', upsample_bilinear2d, '', opset)
    register_op('upsample_trilinear3d', upsample_trilinear3d, '', opset)
    register_op('upsample_bicubic2d', upsample_bicubic2d, '', opset)
