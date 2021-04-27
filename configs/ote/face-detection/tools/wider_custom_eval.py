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

""" This script computes AveragePrecision (VOC) for faces in specific size ranges. """

from argparse import ArgumentParser

from mmcv import DictAction
from ote.metrics.face_detection.custom_voc_ap_eval import custom_voc_ap_evaluation


def parse_args():
    """ Main function. """

    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('config', help='config file path')
    parser.add_argument('input', help='output result file from test.py')
    parser.add_argument('--imsize', nargs=2, type=int, default=(1024, 1024),
                        help='Image resolution. Used for filtering.')
    parser.add_argument('--iou-thr', type=float, default=0.5, help='IoU threshold for evaluation')
    parser.add_argument('--out', help='A path to file where metrics values will be saved (*.json).')
    parser.add_argument('--update_config', nargs='+', action=DictAction,
                        help='Update configuration file by parameters specified here.')
    return parser.parse_args()


custom_voc_ap_evaluation(**vars(parse_args()))
