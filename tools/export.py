# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import argparse
import os.path as osp
import sys
from packaging import version
from subprocess import DEVNULL, CalledProcessError, run

import mmcv
import onnx
import torch
from onnxoptimizer import optimize
from torch.onnx.symbolic_helper import _onnx_stable_opsets as available_opsets

from mmdet.apis import get_fake_input, init_detector
from mmdet.integration.nncf import (check_nncf_is_enabled,
                                    get_nncf_config_from_meta,
                                    is_checkpoint_nncf,
                                    wrap_nncf_model,
                                    get_uncompressed_model)
from mmdet.models import detectors
from mmdet.utils import ExtendedDictAction
from mmdet.utils.deployment.ssd_export_helpers import *  # noqa: F403
from mmdet.utils.deployment.symbolic import (
    register_extra_symbolics, register_extra_symbolics_for_openvino)


def get_min_opset_version():
    return 10 if version.parse(torch.__version__) < version.parse('1.7.0') else 11


def patch_model_for_alt_ssd_export(model):
    model._export_mode = False
    model.onnx_export = onnx_export.__get__(model)
    model.save_img_metas = save_img_metas.__get__(model)
    model.forward = forward.__get__(model)
    model.forward_export = forward_export_detector.__get__(model)
    model.bbox_head.export_forward = export_forward_ssd_head.__get__(model.bbox_head)
    model.bbox_head._prepare_cls_scores_bbox_preds = prepare_cls_scores_bbox_preds_ssd_head.__get__(model.bbox_head)


def patch_nncf_model_for_alt_ssd_export(model):
    model.onnx_export = onnx_export.__get__(model)


def export_to_onnx(model,
                   data,
                   export_name,
                   verbose=False,
                   strip_doc_string=False,
                   opset=10,
                   alt_ssd_export=False,
                   target='onnx'):
    register_extra_symbolics(opset)
    if target == 'openvino' and not alt_ssd_export:
        register_extra_symbolics_for_openvino(opset)

    kwargs = {}
    if torch.__version__ >= '1.5':
        kwargs['enable_onnx_checker'] = False

    if alt_ssd_export:
        assert isinstance(get_uncompressed_model(model), detectors.SingleStageDetector)
        model.onnx_export(img=data['img'][0],
                          img_metas=data['img_metas'][0],
                          export_name=export_name,
                          verbose=verbose,
                          opset_version=opset,
                          strip_doc_string=strip_doc_string,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                          input_names=['image'],
                          output_names=['detection_out'],
                          keep_initializers_as_inputs=True,
                          **kwargs)
    else:
        output_names = ['boxes', 'labels']
        dynamic_axes = {
            'image': {2: 'height', 3: 'width'},
            'boxes': {0: 'objects_num'},
            'labels': {0: 'objects_num'}
        }
        if hasattr(model, 'roi_head'):
            if model.roi_head.with_mask:
                output_names.append('masks')
                dynamic_axes['masks'] = {0: 'objects_num'}
                if getattr(model.roi_head, 'with_text', False):
                    output_names.append('text_features')
                    dynamic_axes['text_features'] = {0: 'objects_num'}

        with torch.no_grad():
            model.export(
                **data,
                f=export_name,
                verbose=verbose,
                opset_version=opset,
                strip_doc_string=strip_doc_string,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                input_names=['image'],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                keep_initializers_as_inputs=True,
                **kwargs
            )


def check_onnx_model(export_name):
    try:
        onnx.checker.check_model(export_name)
        print('ONNX check passed.')
    except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
        print('ONNX check failed.')
        print(ex)


def add_node_names(export_name):
    model = onnx.load(export_name)
    for n in model.graph.node:
        if not n.name:
            n.name = '_'.join([i for i in n.output])
    onnx.save(model, export_name)


def export_to_openvino(cfg, onnx_model_path, output_dir_path, input_shape=None, input_format='bgr', with_text=False):
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    onnx_model = onnx.load(onnx_model_path)
    output_names = set(out.name for out in onnx_model.graph.output)
    # Clear names of the nodes that produce network's output blobs.
    for node in onnx_model.graph.node:
        if output_names.intersection(node.output):
            node.ClearField('name')
    onnx.save(onnx_model, onnx_model_path)
    output_names = ','.join(output_names)

    assert cfg.data.test.pipeline[1]['type'] == 'MultiScaleFlipAug'
    normalize = [v for v in cfg.data.test.pipeline[1]['transforms']
                 if v['type'] == 'Normalize'][0]

    mean_values = normalize['mean']
    scale_values = normalize['std']
    command_line = f'mo.py --input_model="{onnx_model_path}" ' \
                   f'--mean_values="{mean_values}" ' \
                   f'--scale_values="{scale_values}" ' \
                   f'--output_dir="{output_dir_path}" ' \
                   f'--output="{output_names}"'

    assert input_format.lower() in ['bgr', 'rgb']

    if input_shape is not None:
        command_line += f' --input_shape="{input_shape}"'
    if normalize['to_rgb'] and input_format.lower() == 'bgr' or \
            not normalize['to_rgb'] and input_format.lower() == 'rgb':
        command_line += ' --reverse_input_channels'

    print(command_line)

    try:
        run(f'mo.py -h', stdout=DEVNULL, stderr=DEVNULL, shell=True, check=True)
    except CalledProcessError:
        raise RuntimeError('OpenVINO Model Optimizer is not found or configured improperly')

    run(command_line, shell=True, check=True)

    if with_text:
        onnx_model_path_tr_encoder = onnx_model_path.replace('.onnx', '_text_recognition_head_encoder.onnx')
        command_line = f'mo.py --input_model="{onnx_model_path_tr_encoder}" ' \
                       f'--output_dir="{output_dir_path}"'
        print(command_line)
        run(command_line, shell=True, check=True)

        onnx_model_path_tr_decoder = onnx_model_path.replace('.onnx', '_text_recognition_head_decoder.onnx')
        command_line = f'mo.py --input_model="{onnx_model_path_tr_decoder}" ' \
                       f'--output_dir="{output_dir_path}"'
        print(command_line)
        run(command_line, shell=True, check=True)


def optimize_onnx_graph(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)

    onnx_model = optimize(onnx_model, ['extract_constant_to_initializer',
                                       'eliminate_unused_initializer'])

    inputs = onnx_model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in onnx_model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(onnx_model, onnx_model_path)


def main(args):
    assert args.opset in available_opsets
    assert args.opset > 9

    torch.set_default_tensor_type(torch.FloatTensor)

    config = mmcv.Config.fromfile(args.config)
    if args.update_config is not None:
        config.merge_from_dict(args.update_config)
    if args.cfg_options is not None:
        config.merge_from_dict(args.cfg_options)

    model = init_detector(config, args.checkpoint, device='cpu')
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    device = next(model.parameters()).device
    cfg = model.cfg
    fake_data = get_fake_input(cfg, device=device)

    # BEGIN nncf part
    was_model_compressed = is_checkpoint_nncf(args.checkpoint)
    cfg_contains_nncf = cfg.get('nncf_config')

    if cfg_contains_nncf and not was_model_compressed:
        raise RuntimeError('Trying to make export with NNCF compression '
                           'a model snapshot that was NOT trained with NNCF')

    if was_model_compressed and not cfg_contains_nncf:
        # reading NNCF config from checkpoint
        nncf_part = get_nncf_config_from_meta(args.checkpoint)
        for k, v in nncf_part.items():
            cfg[k] = v

    is_alt_ssd_export = getattr(args, 'alt_ssd_export', False)
    if is_alt_ssd_export:
        patch_model_for_alt_ssd_export(model)

    if cfg.get('nncf_config'):
        check_nncf_is_enabled()
        cfg.load_from = args.checkpoint
        cfg.resume_from = None
        compression_ctrl, model = wrap_nncf_model(model, cfg, None, get_fake_input,
                                                  is_alt_ssd_export=is_alt_ssd_export)
        compression_ctrl.prepare_for_export()
        if is_alt_ssd_export:
            patch_nncf_model_for_alt_ssd_export(model)
    # END nncf part

    mmcv.mkdir_or_exist(osp.abspath(args.output_dir))
    onnx_model_path = osp.join(args.output_dir,
                               osp.splitext(osp.basename(args.config))[0] + '.onnx')

    with torch.no_grad():
        export_to_onnx(model, fake_data, export_name=onnx_model_path, opset=args.opset,
                       alt_ssd_export=is_alt_ssd_export,
                       target=args.target, verbose=False)
        add_node_names(onnx_model_path)
        print(f'ONNX model has been saved to "{onnx_model_path}"')

    optimize_onnx_graph(onnx_model_path)

    with_text = False
    if args.target == 'openvino' and not args.alt_ssd_export:
        if hasattr(model, 'roi_head'):
            if getattr(model.roi_head, 'with_text', False):
                with_text = True

    if args.target == 'openvino':
        input_shape = list(fake_data['img'][0].shape)
        if args.input_shape:
            input_shape = [1, 3, *args.input_shape]
        export_to_openvino(cfg, onnx_model_path, args.output_dir, input_shape, args.input_format,
                           with_text=with_text)
    else:
        pass
        # Model check raises a Segmentation Fault in the latest (1.6.0, 1.7.0) versions of onnx package.
        # Even for a valid graph.
        # check_onnx_model(onnx_model_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Export model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help="path to file with model's weights")
    parser.add_argument('output_dir', help='path to directory to save exported models in')
    parser.add_argument('--opset', type=int, default=get_min_opset_version(), help='ONNX opset')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=mmcv.DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--update_config', nargs='+', action=ExtendedDictAction,
                        help='Update configuration file by parameters specified here.')
    subparsers = parser.add_subparsers(title='target', dest='target', help='target model format')
    subparsers.required = True
    parser_onnx = subparsers.add_parser('onnx', help='export to ONNX')
    parser_openvino = subparsers.add_parser('openvino', help='export to OpenVINO')
    parser_openvino.add_argument('--input_shape', nargs=2, type=int, default=None,
                                 help='input shape as a height-width pair')
    parser_openvino.add_argument('--alt_ssd_export', action='store_true',
                                 help='use alternative ONNX representation of SSD net')
    parser_openvino.add_argument('--input_format', choices=['BGR', 'RGB'], default='BGR',
                                 help='Input image format for exported model.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
