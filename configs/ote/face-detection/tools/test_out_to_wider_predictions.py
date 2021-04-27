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

""" This script converts output of test.py (mmdetection) to a set of files
that can be passed to official WiderFace evaluation procedure."""

import argparse

from mmcv import DictAction
from ote.datasets.face_detection.wider_face.convert_predictions import convert_to_wider


def parse_args():
    """ Parses input arguments. """

    parser = argparse.ArgumentParser(
        description='This script converts output of test.py (mmdetection) to '
                    'a set of files that can be passed to official WiderFace '
                    'evaluation procedure.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('input', help='output result file from test.py')
    parser.add_argument('out_folder', help='folder where to store WiderFace '
                                           'evaluation-friendly output')
    parser.add_argument('--update_config', nargs='+', action=DictAction,
                        help='Update configuration file by parameters specified here.')
    args = parser.parse_args()
    return args


convert_to_wider(**vars(parse_args()))
