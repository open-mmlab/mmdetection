# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import pdb
import warnings
from functools import partial

import numpy as np
import onnx
import torch
from mmcv import Config, DictAction

from auto_training.utils.export_specs import export_for_lv
from mmdet.core.export import build_model_from_cfg, preprocess_example_input
from mmdet.core.export.model_wrappers import ONNXRuntimeDetector
from tools.deployment.pytorch2onnx import pytorch2onnx, parse_normalize_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--project_name',
        type=str,
        help='name of the project for lv usage, should match the folder name structure'
             'in the ml_models repo: e.g. \"brummer\"',
        required=True)
    parser.add_argument(
        '--author',
        type=str,
        help='full name of the Author of this training: e.g. \"Christian Holland\"',
        required=True)
    parser.add_argument(
        '--jira_task',
        type=str,
        help='shortened name of the Jira task for this training: e.g. \"OR-1926\"',
        required=True)
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show onnx graph and detection outputs')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--test-img', type=str, default=None, help='Images for test')
    parser.add_argument(
        '--dataset',
        type=str,
        default='coco',
        help='Dataset name. This argument is deprecated and will be removed \
        in future releases.')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=None,
        help='input image size')
    parser.add_argument(
        '--mean',
        type=float,
        nargs='+',
        default=[123.675, 116.28, 103.53],
        help='mean value used for preprocess input data.This argument \
        is deprecated and will be removed in future releases.')
    parser.add_argument(
        '--std',
        type=float,
        nargs='+',
        default=[58.395, 57.12, 57.375],
        help='variance value used for preprocess input data. '
        'This argument is deprecated and will be removed in future releases.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export onnx with dynamic axis.')
    parser.add_argument(
        '--skip-postprocess',
        action='store_true',
        help='Whether to export model without post process. Experimental '
        'option. We do not guarantee the correctness of the exported '
        'model.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    warnings.warn('Arguments like `--mean`, `--std`, `--dataset` would be \
        parsed directly from config file and are deprecated and \
        will be removed in future releases.')

    assert args.opset_version == 11, 'MMDet only support opset 11 now'
    config_path = os.path.join(args.work_dir, "auto_config.py")
    checkpoint_path = os.path.join(args.work_dir, "latest.pth")
    args.config = config_path
    onnx_path = export_for_lv(args)
    try:
        from mmcv.onnx.symbolic import register_extra_symbolics
    except ModuleNotFoundError:
        raise NotImplementedError('please update mmcv to version>=v1.0.4')
    register_extra_symbolics(args.opset_version)


    cfg = Config.fromfile(config_path)
    if args.project_name:
        cfg.project_name = args.project_name
    else:
        cfg.project_name = cfg["data"]["train"]["dataset"]["ann_file"].split("/")[1]

    cfg.author = args.author
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.shape is None:
        img_scale = cfg.test_pipeline[1]['img_scale']
        input_shape = (1, 3, img_scale[1], img_scale[0])
    elif len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    # build the model and load checkpoint
    model = build_model_from_cfg(config_path, checkpoint_path,
                                 args.cfg_options)

    if not args.input_img:
        args.input_img = osp.join(osp.dirname(__file__), '../demo/demo.jpg')

    normalize_cfg = parse_normalize_cfg(cfg.test_pipeline)
    # convert model to onnx file
    pytorch2onnx(
        model,
        args.input_img,
        input_shape,
        normalize_cfg,
        opset_version=args.opset_version,
        show=args.show,
        output_file=onnx_path,
        verify=args.verify,
        test_img=args.test_img,
        do_simplify=args.simplify,
        dynamic_export=args.dynamic_export,
        skip_postprocess=args.skip_postprocess)

    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This tool will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)