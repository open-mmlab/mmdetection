import warnings
from os import path as osp

import onnx
import onnxruntime as ort
import torch

ort_custom_op_path = ''
try:
    from mmcv.ops import get_onnxruntime_op_path
    ort_custom_op_path = get_onnxruntime_op_path()
except (ImportError, ModuleNotFoundError):
    warnings.warn('If input model has custom op from mmcv, \
        you may have to build mmcv with ONNXRuntime from source.')


def verify_model(feat, onnx_io='tmp.onnx'):
    """Run the model in onnxruntime env.

    Args:
        feat (list[Tensor]): A list of tensors from torch.rand,
            each is a 4D-tensor.

    Returns:
        list[np.array]: onnxruntime infer result, each is a np.array
    """

    onnx_model = onnx.load(onnx_io)
    onnx.checker.check_model(onnx_model)

    session_options = ort.SessionOptions()
    # register custom op for onnxruntime
    if osp.exists(ort_custom_op_path):
        session_options.register_custom_ops_library(ort_custom_op_path)
    sess = ort.InferenceSession(onnx_io, session_options)
    if isinstance(feat, torch.Tensor):
        onnx_outputs = sess.run(None,
                                {sess.get_inputs()[0].name: feat.numpy()})
    else:
        onnx_outputs = sess.run(None, {
            sess.get_inputs()[i].name: feat[i].numpy()
            for i in range(len(feat))
        })
    return onnx_outputs


def convert_result_list(outputs):
    """Convert the torch forward outputs containing tuple or list to a list
    only containing torch.Tensor.

    Args:
        output (list(Tensor) | tuple(list(Tensor) | ...): the outputs
        in torch env, maybe containing nested structures such as list
        or tuple.

    Returns:
        list(Tensor): a list only containing torch.Tensor
    """
    # recursive end condition
    if isinstance(outputs, torch.Tensor):
        return [outputs]

    ret = []
    for sub in outputs:
        ret += convert_result_list(sub)
    return ret
