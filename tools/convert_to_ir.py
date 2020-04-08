#!/usr/bin/env python3
#
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

import os

import argparse
import mmcv
import onnx


def main(args):
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    onnx_model = onnx.load(args.input_model)
    output_names = ','.join(out.name for out in onnx_model.graph.output)

    assert cfg.data.test.pipeline[1]['type'] == 'MultiScaleFlipAug'
    normalize = [v for v in cfg.data.test.pipeline[1]['transforms']
                 if v['type'] == 'Normalize'][0]
    shape = cfg.data.test.pipeline[1]['img_scale'][::-1]
    if args.input_shape:
        shape = args.input_shape

    mean_values = normalize['mean']
    scale_values = normalize['std']
    command_line = f'mo.py --input_model="{args.input_model}" ' \
                   f'--mean_values="{mean_values}" ' \
                   f'--scale_values="{scale_values}" ' \
                   f'--output_dir="{args.output_dir}" ' \
                   f'--input_shape="[1,3,{shape[0]},{shape[1]}]" ' \
                   f'--output="{output_names}"' 
    if normalize['to_rgb']:
        command_line += ' --reverse_input_channels'

    print(command_line)
    os.system(command_line)


def parse_args():
    parser = argparse.ArgumentParser(description='Export ONNX model to OpenVINO IR')
    parser.add_argument('config', help='path to configuration file')
    parser.add_argument('input_model', help='path to ONNX model')
    parser.add_argument('output_dir', help='where OpenVINO IR will be saved to')
    parser.add_argument('--input_shape', nargs=2, help='input shape')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
