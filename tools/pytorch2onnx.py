import argparse

import mmcv
import numpy as np
import torch
import torch._C
import torch.serialization
from torch.jit import (_create_interpreter_name_lookup_fn, _flatten,
                       _unique_state_dict)
from torch.onnx import OperatorExportTypes
from torch.onnx.symbolic_helper import (_set_operator_export_type,
                                        _set_opset_version)
from torch.onnx.utils import _optimize_graph
from mmcv.runner import load_checkpoint

from mmdet.models import build_detector


def pytorch2onnx(model,
                 input_shape=(1, 3, 224, 224),
                 opset_version=10,
                 show=False,
                 output_file='tmp.onnx'):
    model.cpu().eval()

    _set_opset_version(opset_version)
    _set_operator_export_type(OperatorExportTypes.ONNX)

    # use dummy input to execute model for tracing
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    one_img = torch.FloatTensor(imgs)
    (_, C, H, W) = input_shape
    one_meta = {
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False
    }

    with torch.no_grad():
        in_vars, _ = _flatten(one_img)
        module_state = list(_unique_state_dict(model, keep_vars=True).values())
        # record the inputs of the graph
        trace, _ = torch._C._tracer_enter(*(in_vars + module_state))
        torch._C._tracer_set_force_outplace(True)
        torch._C._tracer_set_get_unique_name_fn(
            _create_interpreter_name_lookup_fn())
        try:
            # run model for tracing
            tensor_out = model.forward([one_img], [[one_meta]],
                                       return_loss=False)
            out_vars, _ = _flatten(tensor_out)
            # record the outputs of the graph
            torch._C._tracer_exit(tuple(out_vars))
        except Exception:
            torch._C._tracer_abandon()
            raise
        graph = trace.graph()
        params = list(_unique_state_dict(model).values())
        graph = _optimize_graph(graph, OperatorExportTypes.ONNX)
        output_tensors, _ = torch._C._jit_flatten(tensor_out)
        for output, tensor in zip(graph.outputs(), output_tensors):
            output.inferTypeFrom(tensor)
        input_and_param_names = [val.debugName() for val in graph.inputs()]
        param_names = input_and_param_names[len(input_and_param_names) -
                                            len(params):]
        params_dict = dict(zip(param_names, params))
        proto, export_map = graph._export_onnx(params_dict, opset_version, {},
                                               False)
        assert (len(export_map) == 0)
        torch.serialization._with_file_like(output_file, 'wb',
                                            lambda f: f.write(proto))
        if show:
            print(graph)

        return


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMDet to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output_file', type=str, default='tmp.onnx')
    parser.add_argument('--opset_version', type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the model
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    if args.checkpoint:
        checkpoint = load_checkpoint(
            model, args.checkpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

    # conver model to onnx file
    pytorch2onnx(
        model,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file)
