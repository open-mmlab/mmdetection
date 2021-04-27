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

from ote.metrics.face_detection.wider_face.wider_face_eval import wider_face_evaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', required=True)
    parser.add_argument('-g', '--gt', required=True)
    parser.add_argument('--out')

    return parse_args()


args = parse_args()
wider_face_evaluation(args.pred, args.gt, args.out)
