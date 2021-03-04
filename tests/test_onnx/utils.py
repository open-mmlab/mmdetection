import os.path as osp
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch

onnx_io = 'tmp.onnx'

ort_custom_op_path = ''
try:
    from mmcv.ops import get_onnxruntime_op_path
    ort_custom_op_path = get_onnxruntime_op_path()
except (ImportError, ModuleNotFoundError):
    warnings.warn('If input model has custom op from mmcv, \
        you may have to build mmcv with ONNXRuntime from source.')


def verify_model(model, feat):
    """Run the model in the pytorch env and onnxruntime env, and match the
    output of each other.

    Args:
        model (nn.Module): the model which used to run in pytorch and onnx-
            runtime.
        feat (list[Tensor]): A list of tensors from torch.rand to simulate
            input, each is a 4D-tensor.
    """
    torch.onnx.export(
        model,
        feat,
        onnx_io,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=False,
        opset_version=11)

    onnx_model = onnx.load(onnx_io)
    onnx.checker.check_model(onnx_model)

    session_options = ort.SessionOptions()
    # register custom op for onnxruntime
    if osp.exists(ort_custom_op_path):
        session_options.register_custom_ops_library(ort_custom_op_path)
    sess = ort.InferenceSession(onnx_io, session_options)
    onnx_outputs = sess.run(
        None,
        {sess.get_inputs()[i].name: feat[i].numpy()
         for i in range(len(feat))})

    torch_outputs = model.forward(feat)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)
