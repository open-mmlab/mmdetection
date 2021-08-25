from functools import wraps

import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_registry import get_registered_op, is_registered_op

DOMAIN_CUSTOM_OPS_NAME = 'org.openvinotoolkit'


def add_domain(name_operator: str) -> str:
    return DOMAIN_CUSTOM_OPS_NAME + '::' + name_operator


def py_symbolic(op_name=None, namespace='mmdet_custom', adapter=None):
    """The py_symbolic decorator allows associating a function with a custom
    symbolic function that defines its representation in a computational graph.

    A symbolic function cannot receive a collection of tensors as arguments.
    If your custom function takes a collection of tensors as arguments,
    then you need to implement an argument converter (adapter)
    from the collection and pass it to the decorator.

    Args:
        op_name (str): Operation name, must match
                       the registered operation name.
        namespace (str): Namespace for this operation.
        adapter (function): Function for converting arguments.

    Adapter conventions:
        1. The adapter must have the same signature as the wrapped function.
        2. The values, returned by the adapter, must match
           the called symbolic function.
        3. Return value order:
            tensor values (collections are not supported)
            constant parameters (can be passed using a dictionary)

    Usage example:
        1. Implement a custom operation. For example 'custom_op'.
        2. Implement a symbolic function to represent the custom_op in
            a computation graph. For example 'custom_op_symbolic'.
        3. Register the operation before export:
            register_op('custom_op_name', custom_op_symbolic, namespace, opset)
        4. Decorate the custom operation:
            @py_symbolic(op_name='custom_op_name')
            def custom_op(...):
        5. If you need to convert custom function arguments to symbolic
           function arguments, you can implement a converter and
           pass it to the decorator:
            @py_symbolic(op_name='custom_op_name', adapter=converter)
    """

    def decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):

            name = op_name if op_name is not None else func.__name__
            opset = sym_help._export_onnx_opset_version

            if is_registered_op(name, namespace, opset):

                class XFunction(torch.autograd.Function):

                    @staticmethod
                    def forward(ctx, *xargs):
                        return func(*args, **kwargs)

                    @staticmethod
                    def symbolic(g, *xargs):
                        symb = get_registered_op(name, namespace, opset)
                        if adapter is not None:
                            return symb(g, *xargs, **adapter_kwargs)
                        return symb(g, *xargs)

                if adapter is not None:
                    adapter_args, adapter_kwargs = adapter(*args, **kwargs)
                    return XFunction.apply(*adapter_args)
                return XFunction.apply(*args)
            else:
                return func(*args, **kwargs)

        return wrapped_function

    return decorator


class PatchSymbolic():
    """
    This class is the standard representation of a symbolic patch.
    Contains:
        the name of the operation,
        the symbolic function,
        function that applies the changes.

    To automatically add a new patch,
    you need to create the "get_patch_" function,
    which will return the "PatchSymbolic" object.
    """

    def __init__(self, operation_name, symbolic_func, patch_func):
        self.operation_name = operation_name
        self.symbolic_func = symbolic_func
        self.patch_func = patch_func

    def apply_patch(self):
        self.patch_func()

    def get_symbolic_func(self):
        return self.symbolic_func

    def get_operation_name(self):
        return self.operation_name


def get_patch_roi_feature_extractor() -> PatchSymbolic:
    """Replaces standard RoiAlign with ExperimentalDetectronROIFeatureExtractor
    for faster work in OpenVINO IR.

    Used in Faster-RCNN, Mask-RCNN, Cascade-RCNN, Cascade-Mask-RCNN.
    """
    operation_name = 'roi_feature_extractor'

    def symbolic(g,
                 rois,
                 *feats,
                 output_size=1,
                 featmap_strides=1,
                 sample_num=1):
        from torch.onnx.symbolic_helper import _slice_helper
        rois = _slice_helper(g, rois, axes=[1], starts=[1], ends=[5])
        roi_feats = g.op(
            add_domain('ExperimentalDetectronROIFeatureExtractor'),
            rois,
            *feats,
            output_size_i=output_size,
            pyramid_scales_i=featmap_strides,
            sampling_ratio_i=sample_num,
            image_id_i=0,
            distribute_rois_between_levels_i=1,
            preserve_rois_order_i=0,
            aligned_i=1,
            outputs=1)
        return roi_feats

    def patch():

        def adapter(self, feats, rois):
            return ((rois, ) + tuple(feats), {
                'output_size': self.roi_layers[0].output_size[0],
                'featmap_strides': self.featmap_strides,
                'sample_num': self.roi_layers[0].sampling_ratio
            })

        from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor\
            import SingleRoIExtractor
        SingleRoIExtractor.forward = \
            py_symbolic(op_name=operation_name,
                        adapter=adapter)(SingleRoIExtractor.forward)

    return PatchSymbolic(operation_name, symbolic, patch)
