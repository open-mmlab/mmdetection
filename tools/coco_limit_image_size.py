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
import json
import os
from math import floor

import cv2
from tqdm import tqdm


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('annotation', help='Path to COCO annotation (*.json).')
    args.add_argument('input_root', help='Path to input images root folder.')
    args.add_argument('output_root', help='Path to output images root folder.')
    args.add_argument('--limit_image_size', type=int, nargs=2, default=(1080, 1920),
                      help='Resize images with size greater than specified here size (h, w)! '
                           'keeping aspect ratio.')
    return args.parse_args()


def main():
    args = parse_args()
    assert args.input_root != args.output_root
    assert not os.path.exists(args.output_root)

    with open(args.annotation) as f:
        content = json.load(f)

    resized = 0

    for i, image_info in enumerate(tqdm(content['images'])):
        path = os.path.join(args.input_root, image_info['file_name'])
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f'Failed to read image: {path}.')

        height_ratio = image.shape[0] / args.limit_image_size[0]
        width_ratio = image.shape[1] / args.limit_image_size[1]
        if height_ratio > 1 or width_ratio > 1:
            ratio = height_ratio if height_ratio > width_ratio else width_ratio
            new_size = int(floor(image.shape[1] / ratio)), int(floor(image.shape[0] / ratio))
            image = cv2.resize(image, new_size)
            resized += 1

        output_path = os.path.join(args.output_root, image_info['file_name'])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)

    print(f'Resized images: {resized} of {len(content["images"])}')


if __name__ == '__main__':
    main()
