import argparse
import os
import os.path as osp

import mmcv
import numpy as np
import onnx
import torch
from mmcv import Config
from mmcv.tensorrt import is_tensorrt_plugin_loaded, onnx2trt, save_trt_engine

from mmdet.core import get_classes
from mmdet.core.export import prepare_inputs
from mmdet.core.export.model_wrappers import (ONNXRuntimeDetector,
                                              TensorRTDetector)


def get_GiB(x: int):
    """return x GiB."""
    return x * (1 << 30)


def onnx2tensorrt(onnx_file,
                  trt_file,
                  input_img,
                  input_shape,
                  test_pipeline,
                  verify=False,
                  show=False,
                  dataset='coco',
                  workspace_size=1,
                  verbose=False):
    import tensorrt as trt
    onnx_model = onnx.load(onnx_file)
    # create trt engine and wraper
    opt_shape_dict = {'input': [input_shape, input_shape, input_shape]}
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
        # prepare inputs
        img_list, img_meta_list = prepare_inputs(
            input_img,
            test_pipeline,
            shape=input_shape[2:],
            keep_ratio=False,
            rescale=True)
        img_list = [_.cuda().contiguous() for _ in img_list]

        # wrap ONNX and TensorRT model
        CLASSES = get_classes(dataset)
        onnx_model = ONNXRuntimeDetector(onnx_file, CLASSES, device_id=0)
        trt_model = TensorRTDetector(trt_file, CLASSES, device_id=0)

        # inference with wrapped model
        with torch.no_grad():
            onnx_results = onnx_model(
                img_list, img_metas=img_meta_list, return_loss=False)[0]
            trt_results = trt_model(
                img_list, img_metas=img_meta_list, return_loss=False)[0]

        if show:
            # visualize
            from mmdet.apis import show_result_pyplot
            show_img = mmcv.imread(input_img)
            show_result_pyplot(
                onnx_model, show_img, onnx_results, title='ONNXRuntime')
            show_result_pyplot(
                trt_model, show_img, trt_results, title='TensorRT')

        with_mask = trt_model.with_masks
        # compare a part of result
        if with_mask:
            compare_pairs = list(zip(onnx_results, trt_results))
        else:
            compare_pairs = [(onnx_results, trt_results)]
        err_msg = 'The numerical values are different between Pytorch' + \
                  ' and ONNX, but it does not necessarily mean the' + \
                  ' exported ONNX model is problematic.'
        # check the numerical value
        for onnx_res, pytorch_res in compare_pairs:
            for o_res, p_res in zip(onnx_res, pytorch_res):
                np.testing.assert_allclose(
                    o_res, p_res, rtol=1e-03, atol=1e-05, err_msg=err_msg)
        print('The numerical values are the same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models from ONNX to TensorRT')
    parser.add_argument('config', help='test config file path')
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
        '--shape',
        type=int,
        nargs='+',
        default=[400, 600],
        help='Input size of the model')
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

    cfg = Config.fromfile(args.config)

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    # Create TensorRT engine
    onnx2tensorrt(
        args.model,
        args.trt_file,
        args.input_img,
        input_shape,
        cfg.test_pipeline,
        verify=args.verify,
        show=args.show,
        dataset=args.dataset,
        workspace_size=args.workspace_size,
        verbose=args.verbose)
