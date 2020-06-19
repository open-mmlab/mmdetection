"""Modified from https://github.com/pytorch/pytorch."""
from __future__ import absolute_import, division, print_function
import warnings
from functools import wraps
from sys import maxsize as maxsize

import torch
import torch.onnx
# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
import torch.onnx.utils
from torch._C import ListType

# ---------------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------------

# Save some builtins as locals, because we'll shadown them below
_sum = sum


def _parse_arg(value, desc):
    if desc == 'none':
        return value
    if desc == 'v' or not _is_value(value):
        return value
    if value.node().mustBeNone():
        return None
    if value.node().kind() == 'onnx::Constant':
        tval = value.node()['value']
        if desc == 'i':
            return int(tval)
        elif desc == 'f':
            return float(tval)
        elif desc == 'b':
            return bool(tval)
        elif desc == 's':
            return str(tval)
        elif desc == 't':
            return tval
        elif desc == 'is':
            return [int(v) for v in tval]
        else:
            raise RuntimeError(
                "ONNX symbolic doesn't know to interpret Constant node")
    elif value.node().kind() == 'prim::ListConstruct':
        if desc == 'is':
            for v in value.node().inputs():
                if v.node().kind() != 'onnx::Constant':
                    raise RuntimeError(
                        "Failed to export an ONNX attribute '" +
                        v.node().kind() +
                        "', since it's not constant, please try to make "
                        'things (e.g., kernel size) static if possible')
            return [int(v.node()['value']) for v in value.node().inputs()]
        else:
            raise RuntimeError(
                "ONNX symbolic doesn't know to interpret ListConstruct node")

    raise RuntimeError('Unexpected node type: {}'.format(value.node().kind()))


def _maybe_get_const(value, desc):
    if _is_value(value) and value.node().kind() == 'onnx::Constant':
        return _parse_arg(value, desc)
    return value


def _maybe_get_scalar(value):
    value_t = _maybe_get_const(value, 't')
    if isinstance(value_t, torch.Tensor) and value_t.shape == ():
        return value_t
    return value


def _get_const(value, desc, arg_name):
    if _is_value(value) and value.node().kind() not in ('onnx::Constant',
                                                        'prim::Constant'):
        raise RuntimeError('ONNX symbolic expected a constant'
                           ' value of the {} argument, got `{}`'.format(
                               arg_name, value))
    return _parse_arg(value, desc)


def _unpack_list(list_value):
    list_node = list_value.node()
    assert list_node.kind() == 'prim::ListConstruct'
    return list(list_node.inputs())


# Check if list_value is output from prim::ListConstruct
# This is usually called before _unpack_list to ensure the list can be
# unpacked.
def _is_packed_list(list_value):
    return _is_value(
        list_value) and list_value.node().kind() == 'prim::ListConstruct'


def parse_args(*arg_descriptors):

    def decorator(fn):
        fn._arg_descriptors = arg_descriptors

        def wrapper(g, *args):
            # some args may be optional, so the length may be smaller
            assert len(arg_descriptors) >= len(args)
            args = [
                _parse_arg(arg, arg_desc)
                for arg, arg_desc in zip(args, arg_descriptors)
            ]
            return fn(g, *args)

        # In Python 2 functools.wraps chokes on partially applied functions, so
        # we need this as a workaround
        try:
            wrapper = wraps(fn)(wrapper)
        except Exception:
            pass
        return wrapper

    return decorator


def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x.item()


def _if_scalar_type_as(g, self, tensor):
    """Convert self into the same type of tensor, as necessary.

    We only support implicit casting for scalars, so we never actually
    need to insert an ONNX cast operator here; just fix up the scalar.
    """
    if isinstance(self, torch._C.Value):
        return self

    scalar_type = tensor.type().scalarType()
    if scalar_type:
        ty = scalar_type.lower()
        return getattr(self, ty)()

    return self


def _is_none(x):
    return x.node().mustBeNone()


def _is_value(x):
    return isinstance(x, torch._C.Value)


def _is_tensor_list(x):
    return x.type().isSubtypeOf(ListType.ofTensors())


def _unimplemented(op, msg):
    warnings.warn('ONNX export failed on ' + op + ' because ' + msg +
                  ' not supported')


def _try_get_scalar_type(*args):
    for arg in args:
        try:
            return arg.type().scalarType()
        except RuntimeError:
            pass
    return None


def _topk_helper(g, input, k, dim, largest=True, sorted=False, out=None):
    if out is not None:
        _unimplemented('TopK', 'Out parameter is not supported')
    if not _is_value(k):
        k = g.op('Constant', value_t=torch.tensor([k], dtype=torch.int64))
    else:
        k = g.op('Reshape', k, g.op('Constant', value_t=torch.tensor([1])))
    return g.op(
        'TopK',
        input,
        k,
        axis_i=dim,
        largest_i=largest,
        sorted_i=sorted,
        outputs=2)


def _slice_helper(g,
                  input,
                  axes,
                  starts,
                  ends,
                  steps=None,
                  dynamic_slice=False):
    # TODO(ruobing): add support for opset<10
    from torch.onnx.symbolic_opset10 import _slice
    return _slice(g, input, axes, starts, ends, steps, dynamic_slice)


def _unsqueeze_helper(g, input, dim):
    from torch.onnx.symbolic_opset9 import unsqueeze
    return unsqueeze(g, input, dim)


def _interpolate_size_to_scales(g, input, output_size, dim):
    output_size = _maybe_get_const(output_size, 'is')
    if _is_value(output_size):
        offset = 2
        offsets = g.op(
            'Constant', value_t=torch.ones(offset, dtype=torch.float32))
        dividend = g.op(
            'Cast', output_size, to_i=cast_pytorch_to_onnx['Float'])
        divisor = _slice_helper(
            g, g.op('Shape', input), axes=[0], ends=[maxsize], starts=[offset])
        divisor = g.op('Cast', divisor, to_i=cast_pytorch_to_onnx['Float'])
        scale_dims = g.op('Div', dividend, divisor)
        scales = g.op('Concat', offsets, scale_dims, axis_i=0)
    else:
        scales_constant = [
            1. if i < 2 else float(output_size[-(dim - i)]) /
            float(input.type().sizes()[-(dim - i)]) for i in range(0, dim)
        ]
        scales = g.op(
            'Constant',
            value_t=torch.tensor(scales_constant, dtype=torch.float32))
    return scales


def _interpolate_get_scales_if_available(g, scales):
    if len(scales) == 0:
        return None
    available_scales = _maybe_get_const(scales[0], 'f') != -1 and not _is_none(
        scales[0])

    if not available_scales:
        return None

    scales_list = []
    for scale in scales:
        unsqueezed_scale = _unsqueeze_helper(g, scale, 0)
        # ONNX only supports float for the scales. double -> float.
        unsqueezed_scale = g.op(
            'Cast', unsqueezed_scale, to_i=cast_pytorch_to_onnx['Float'])
        scales_list.append(unsqueezed_scale)
    offsets = g.op('Constant', value_t=torch.ones(2, dtype=torch.float32))
    scales = g.op('Concat', offsets, *scales_list, axis_i=0)
    return scales


def _get_interpolate_attributes(g, mode, args):
    if mode == 'nearest':
        align_corners = None
        scales = args[0:]
    else:
        align_corners = args[0]
        scales = args[1:]
    scales = _interpolate_get_scales_if_available(g, scales)
    return scales, align_corners


def _interpolate_get_scales(g, scale_factor, dim):
    offsets = g.op('Constant', value_t=torch.ones(2, dtype=torch.float32))
    if isinstance(scale_factor.type(), torch._C.ListType):
        return g.op('Concat', offsets, scale_factor, axis_i=0)
    else:
        scale_factor = _unsqueeze_helper(g, scale_factor, 0)
        scale_factor = g.op(
            'Cast', scale_factor, to_i=cast_pytorch_to_onnx['Float'])
        scales = [scale_factor for i in range(dim - 2)]
    scale_factor = g.op('Concat', offsets, *scales, axis_i=0)
    return scale_factor


# Metaprogram symbolics for each ATen native specialized cast operator.
# For e.g. we specify a function named `_cast_uint8_t` that instantiates an
# ONNX cast node with `to` attribute 'UINT8'
#
# TODO: remove these once we support Type's in the JIT IR and we can once again
# use the unified toType operator
cast_pytorch_to_onnx = {
    'Byte': torch.onnx.TensorProtoDataType.UINT8,
    'Char': torch.onnx.TensorProtoDataType.INT8,
    'Double': torch.onnx.TensorProtoDataType.DOUBLE,
    'Float': torch.onnx.TensorProtoDataType.FLOAT,
    'Half': torch.onnx.TensorProtoDataType.FLOAT16,
    'Int': torch.onnx.TensorProtoDataType.INT32,
    'Long': torch.onnx.TensorProtoDataType.INT64,
    'Short': torch.onnx.TensorProtoDataType.INT16,
    'Bool': torch.onnx.TensorProtoDataType.BOOL,
    'ComplexFloat': torch.onnx.TensorProtoDataType.COMPLEX64,
    'ComplexDouble': torch.onnx.TensorProtoDataType.COMPLEX128,
    'Undefined': torch.onnx.TensorProtoDataType.UNDEFINED,
}

# Global set to store the list of quantized operators in the network.
# This is currently only used in the conversion of quantized ops from PT
# -> C2 via ONNX.
_quantized_ops = set()
