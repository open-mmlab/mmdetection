# Copyright (C) 2021 Intel Corporation
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
import sys

import mmcv
import torch

from mmdet.apis import get_fake_input, init_detector
from mmdet.apis.export import export_model, patch_model_for_alt_ssd_export, patch_nncf_model_for_alt_ssd_export
from mmdet.integration.nncf import (check_nncf_is_enabled,
                                    get_nncf_config_from_meta,
                                    is_checkpoint_nncf,
                                    wrap_nncf_model)
from mmdet.utils import ExtendedDictAction
from mmdet.utils.deployment.ssd_export_helpers import *  # noqa: F403


def main(args):
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

    is_alt_ssd_export = getattr(args, 'alt_ssd_export', False)
    if is_alt_ssd_export:
        patch_model_for_alt_ssd_export(model)

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

    if args.target == 'onnx':
        export_model(model, cfg, args.output_dir, target=args.target, onnx_opset=args.opset)
    else:
        export_model(model, cfg, args.output_dir, target=args.target, onnx_opset=args.opset,
                    input_shape=args.input_shape, input_format=args.input_format, precision=args.precision,
                    alt_ssd_export=is_alt_ssd_export)


def parse_args():
    parser = argparse.ArgumentParser(description='Export model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help="path to file with model's weights")
    parser.add_argument('output_dir', help='path to directory to save exported models in')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset')
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
    parser_openvino.add_argument('--precision', choices=['FP16', 'FP32'], default='FP32',
                                 help='Numerical precision of output models\' weights.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
