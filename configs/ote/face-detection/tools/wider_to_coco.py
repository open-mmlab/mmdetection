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

""" Converts WiderFace annotation to COCO format. """

import argparse

from ote.datasets.face_detection.wider_face.convert_annotation import convert_to_coco


def parse_args():
    """ Parses input arguments. """

    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('input_annotation',
                        help="Path to annotation file like wider_face_train_bbx_gt.txt")
    parser.add_argument('images_dir',
                        help="Path to folder with images like WIDER_train/images")
    parser.add_argument('output_annotation', help="Path to output json file")
    parser.add_argument('--with_landmarks', action='store_true',
                        help="Whether to read landmarks")

    return parser.parse_args()


args = parse_args()
convert_to_coco(args.input_annotation, args.images_dir,
                args.output_annotation, args.with_landmarks)
