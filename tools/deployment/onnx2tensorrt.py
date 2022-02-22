# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings

import numpy as np
import onnx
import torch
from mmcv import Config
from mmcv.tensorrt import is_tensorrt_plugin_loaded, onnx2trt, save_trt_engine

from mmdet.core.export import preprocess_example_input
from mmdet.core.export.model_wrappers import (ONNXRuntimeDetector,
                                              TensorRTDetector)
from mmdet.datasets import DATASETS


def get_GiB(x: int):
    """return x GiB."""
    return x * (1 << 30)


def onnx2tensorrt(onnx_file,
                  trt_file,
                  input_config,
                  verify=False,
                  show=False,
                  workspace_size=1,
                  verbose=False):
    import tensorrt as trt
    onnx_model = onnx.load(onnx_file)
    max_shape = input_config['max_shape']
    min_shape = input_config['min_shape']
    opt_shape = input_config['opt_shape']
    fp16_mode = False
    # create trt engine and wrapper
    opt_shape_dict = {'input': [min_shape, opt_shape, max_shape]}
    max_workspace_size = get_GiB(workspace_size)
    trt_engine = onnx2trt(
        onnx_model,
        opt_shape_dict,
        log_level=trt.Logger.VERBOSE if verbose else trt.Logger.ERROR,
        fp16_mode=fp16_mode,
        max_workspace_size=max_workspace_size)
    save_dir, _ = osp.split(trt_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    save_trt_engine(trt_engine, trt_file)
    print(f'Successfully created TensorRT engine: {trt_file}')

    if verify:
        # prepare input
        one_img, one_meta = preprocess_example_input(input_config)
        img_list, img_meta_list = [one_img], [[one_meta]]
        img_list = [_.cuda().contiguous() for _ in img_list]

        # wrap ONNX and TensorRT model
        onnx_model = ONNXRuntimeDetector(onnx_file, CLASSES, device_id=0)
        trt_model = TensorRTDetector(trt_file, CLASSES, device_id=0)

        # inference with wrapped model
        with torch.no_grad():
            onnx_results = onnx_model(
                img_list, img_metas=img_meta_list, return_loss=False)[0]
            trt_results = trt_model(
                img_list, img_metas=img_meta_list, return_loss=False)[0]

        if show:
            out_file_ort, out_file_trt = None, None
        else:
            out_file_ort, out_file_trt = 'show-ort.png', 'show-trt.png'
        show_img = one_meta['show_img']
        score_thr = 0.3
        onnx_model.show_result(
            show_img,
            onnx_results,
            score_thr=score_thr,
            show=True,
            win_name='ONNXRuntime',
            out_file=out_file_ort)
        trt_model.show_result(
            show_img,
            trt_results,
            score_thr=score_thr,
            show=True,
            win_name='TensorRT',
            out_file=out_file_trt)
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


def parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    assert len(norm_config_li) == 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0]
    return norm_config


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
        '--dataset',
        type=str,
        default='coco',
        help='Dataset name. This argument is deprecated and will be \
        removed in future releases.')
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
        help='Feed model with RGB or BGR image. Default is RGB. This \
        argument is deprecated and will be removed in future releases.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[400, 600],
        help='Input size of the model')
    parser.add_argument(
        '--mean',
        type=float,
        nargs='+',
        default=[123.675, 116.28, 103.53],
        help='Mean value used for preprocess input data. This argument \
        is deprecated and will be removed in future releases.')
    parser.add_argument(
        '--std',
        type=float,
        nargs='+',
        default=[58.395, 57.12, 57.375],
        help='Variance value used for preprocess input data. \
        This argument is deprecated and will be removed in future releases.')
    parser.add_argument(
        '--min-shape',
        type=int,
        nargs='+',
        default=None,
        help='Minimum input size of the model in TensorRT')
    parser.add_argument(
        '--max-shape',
        type=int,
        nargs='+',
        default=None,
        help='Maximum input size of the model in TensorRT')
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
    warnings.warn(
        'Arguments like `--to-rgb`, `--mean`, `--std`, `--dataset` would be \
        parsed directly from config file and are deprecated and will be \
        removed in future releases.')
    if not args.input_img:
        args.input_img = osp.join(osp.dirname(__file__), '../../demo/demo.jpg')

    cfg = Config.fromfile(args.config)

    def parse_shape(shape):
        if len(shape) == 1:
            shape = (1, 3, shape[0], shape[0])
        elif len(args.shape) == 2:
            shape = (1, 3) + tuple(shape)
        else:
            raise ValueError('invalid input shape')
        return shape

    if args.shape:
        input_shape = parse_shape(args.shape)
    else:
        img_scale = cfg.test_pipeline[1]['img_scale']
        input_shape = (1, 3, img_scale[1], img_scale[0])

    if not args.max_shape:
        max_shape = input_shape
    else:
        max_shape = parse_shape(args.max_shape)

    if not args.min_shape:
        min_shape = input_shape
    else:
        min_shape = parse_shape(args.min_shape)

    dataset = DATASETS.get(cfg.data.test['type'])
    assert (dataset is not None)
    CLASSES = dataset.CLASSES
    normalize_cfg = parse_normalize_cfg(cfg.test_pipeline)

    input_config = {
        'min_shape': min_shape,
        'opt_shape': input_shape,
        'max_shape': max_shape,
        'input_shape': input_shape,
        'input_path': args.input_img,
        'normalize_cfg': normalize_cfg
    }
    # Create TensorRT engine
    onnx2tensorrt(
        args.model,
        args.trt_file,
        input_config,
        verify=args.verify,
        show=args.show,
        workspace_size=args.workspace_size,
        verbose=args.verbose)
