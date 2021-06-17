import argparse
import os
import os.path as osp

import numpy as np
import onnx
import onnxruntime as ort
import torch
from mmcv.ops import get_onnxruntime_op_path
from mmcv.tensorrt import (TRTWraper, is_tensorrt_plugin_loaded, onnx2trt,
                           save_trt_engine)

from mmdet.core import get_classes
from mmdet.core.export import preprocess_example_input
from mmdet.core.visualization.image import imshow_det_bboxes


def get_GiB(x: int):
    """return x GiB."""
    return x * (1 << 30)


def onnx2tensorrt(onnx_file,
                  trt_file,
                  input_config,
                  verify=False,
                  show=False,
                  dataset='coco',
                  workspace_size=1,
                  verbose=False):
    import tensorrt as trt
    onnx_model = onnx.load(onnx_file)
    input_shape = input_config['input_shape']
    max_shape = input_config['max_shape']
    # create trt engine and wraper
    opt_shape_dict = {'input': [input_shape, input_shape, max_shape]}
    max_workspace_size = get_GiB(workspace_size)
    trt_engine = onnx2trt(
        onnx_model,
        opt_shape_dict,
        log_level=trt.Logger.VERBOSE if verbose else trt.Logger.ERROR,
        fp16_mode=False,
        max_workspace_size=max_workspace_size)
    save_dir, _ = osp.split(trt_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    save_trt_engine(trt_engine, trt_file)
    print(f'Successfully created TensorRT engine: {trt_file}')

    if verify:
        one_img, one_meta = preprocess_example_input(input_config)
        input_img_cpu = one_img.detach().cpu().numpy()
        input_img_cuda = one_img.cuda()
        img = one_meta['show_img']

        # Get results from ONNXRuntime
        ort_custom_op_path = get_onnxruntime_op_path()
        session_options = ort.SessionOptions()
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        sess = ort.InferenceSession(onnx_file, session_options)
        output_names = [_.name for _ in sess.get_outputs()]
        ort_outputs = sess.run(None, {
            'input': input_img_cpu,
        })
        with_mask = len(output_names) == 3
        ort_outputs = [_.squeeze(0) for _ in ort_outputs]
        ort_dets, ort_labels = ort_outputs[:2]
        ort_masks = ort_outputs[2] if with_mask else None
        ort_shapes = [_.shape for _ in ort_outputs]
        print(f'ONNX Runtime output names: {output_names}, \
            output shapes: {ort_shapes}')

        # Get results from TensorRT
        trt_model = TRTWraper(trt_file, ['input'], output_names)
        with torch.no_grad():
            trt_outputs = trt_model({'input': input_img_cuda})
        trt_outputs = [
            trt_outputs[_].detach().cpu().numpy().squeeze(0)
            for _ in output_names
        ]
        trt_dets, trt_labels = trt_outputs[:2]
        trt_shapes = [_.shape for _ in trt_outputs]
        print(f'TensorRT output names: {output_names}, \
            output shapes: {trt_shapes}')
        trt_masks = trt_outputs[2] if with_mask else None

        if trt_masks is not None and trt_masks.dtype != np.bool:
            trt_masks = trt_masks >= 0.5
            ort_masks = ort_masks >= 0.5
        # Show detection outputs
        if show:
            CLASSES = get_classes(dataset)
            score_thr = 0.35
            imshow_det_bboxes(
                img.copy(),
                trt_dets,
                trt_labels,
                segms=trt_masks,
                class_names=CLASSES,
                score_thr=score_thr,
                win_name='TensorRT')
            imshow_det_bboxes(
                img.copy(),
                ort_dets,
                ort_labels,
                segms=ort_masks,
                class_names=CLASSES,
                score_thr=score_thr,
                win_name='ONNXRuntime')
        # Compare results
        np.testing.assert_allclose(ort_dets, trt_dets, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(ort_labels, trt_labels)
        if with_mask:
            np.testing.assert_allclose(
                ort_masks, trt_masks, rtol=1e-03, atol=1e-05)
        print('The numerical values are the same ' +
              'between ONNXRuntime and TensorRT')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models from ONNX to TensorRT')
    parser.add_argument('model', help='Filename of input ONNX model')
    parser.add_argument(
        '--trt-file',
        type=str,
        default='tmp.trt',
        help='Filename of output TensorRT engine')
    parser.add_argument(
        '--input-img', type=str, default='', help='Image for test')
    parser.add_argument(
        '--show', action='store_true', help='Whether to show output results')
    parser.add_argument(
        '--dataset', type=str, default='coco', help='Dataset name')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the outputs of ONNXRuntime and TensorRT')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether to verbose logging messages while creating \
                TensorRT engine. Defaults to False.')
    parser.add_argument(
        '--to-rgb',
        action='store_false',
        help='Feed model with RGB or BGR image. Default is RGB.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[400, 600],
        help='Input size of the model')
    parser.add_argument(
        '--max-shape',
        type=int,
        nargs='+',
        default=None,
        help='Maximum input size of the model in TensorRT')
    parser.add_argument(
        '--mean',
        type=float,
        nargs='+',
        default=[123.675, 116.28, 103.53],
        help='Mean value used for preprocess input data')
    parser.add_argument(
        '--std',
        type=float,
        nargs='+',
        default=[58.395, 57.12, 57.375],
        help='Variance value used for preprocess input data')
    parser.add_argument(
        '--workspace-size',
        type=int,
        default=1,
        help='Max workspace size in GiB')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    assert is_tensorrt_plugin_loaded(), 'TensorRT plugin should be compiled.'
    args = parse_args()

    if not args.input_img:
        args.input_img = osp.join(osp.dirname(__file__), '../demo/demo.jpg')

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    if not args.max_shape:
        max_shape = input_shape
    else:
        if len(args.max_shape) == 1:
            max_shape = (1, 3, args.max_shape[0], args.max_shape[0])
        elif len(args.max_shape) == 2:
            max_shape = (1, 3) + tuple(args.max_shape)
        else:
            raise ValueError('invalid input max_shape')

    assert len(args.mean) == 3
    assert len(args.std) == 3

    normalize_cfg = {'mean': args.mean, 'std': args.std, 'to_rgb': args.to_rgb}
    input_config = {
        'input_shape': input_shape,
        'input_path': args.input_img,
        'normalize_cfg': normalize_cfg,
        'max_shape': max_shape
    }

    # Create TensorRT engine
    onnx2tensorrt(
        args.model,
        args.trt_file,
        input_config,
        verify=args.verify,
        show=args.show,
        dataset=args.dataset,
        workspace_size=args.workspace_size,
        verbose=args.verbose)
