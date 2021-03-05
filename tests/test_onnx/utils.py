import warnings
from os import getcwd
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
    """Run the model in the pytorch env and onnxruntime env, and match the
    output of each other.

    Args:
        model (nn.Module): the model which used to run in pytorch and onnx-
            runtime.
        feat (list[Tensor]): A list of tensors from torch.rand to simulate
            input, each is a 4D-tensor.
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


def list_gen(outputs):
    while True:
        ret = []
        flags = True
        for i in outputs:
            if not isinstance(i, torch.Tensor):
                flags = False
                for j in i:
                    ret.append(j)
            else:
                ret.append(i)
        outputs = ret
        if flags:
            break
    return outputs


def get_data_path():
    exe_path = getcwd().split('/')[-1]
    if exe_path == 'tests':
        data_path = osp.join(getcwd(), 'test_onnx/data/')
    elif exe_path == 'test_onnx':
        data_path = osp.join(getcwd(), 'data/')
    else:
        data_path = osp.join(getcwd(), 'tests/test_onnx/data/')
    return data_path
