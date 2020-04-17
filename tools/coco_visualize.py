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

''' This scrip visualizes bounding boxes from COCO annotation. '''

import argparse
import json
import os
import random
from collections import defaultdict
from math import floor

import cv2
from tqdm import tqdm


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('annotation', help='Path to COCO annotation (*.json).')
    args.add_argument('root', help='Path to images root folder.')
    args.add_argument('--delay', type=int, default=0, help='imshow delay.')
    args.add_argument('--limit_imshow_size', type=int, nargs=2, default=(720, 1280),
                      help='Resize images with size greater than specified here size (h, w)! '
                           'keeping aspect ratio.')
    args.add_argument('--shuffle', action='store_true',
                      help='Shuffle annotation before visualization.')

    return args.parse_args()


def print_stat(content):
    print('   images:', len(content['images']))
    print('   annotations:', len(content['annotations']))
    print('   categories:')
    for cat in content['categories']:
        print('      ', cat)


def main():
    args = parse_args()

    with open(args.annotation) as f:
        content = json.load(f)

    print_stat(content)

    annotations = defaultdict(list)
    categories = {}

    for ann in content['annotations']:
        annotations[ann['image_id']].append(ann)

    for cat in content['categories']:
        categories[cat['id']] = cat

    if args.shuffle:
        random.shuffle(content['images'])

    for image_info in tqdm(content['images']):
        path = os.path.join(args.root, image_info['file_name'])
        image = cv2.imread(path)
        if image is None:
            print(path)

        for ann in annotations[image_info['id']]:
            bbox = ann['bbox']
            p1 = int(bbox[0]), int(bbox[1])
            p2 = int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])

            try:
                cv2.putText(image, categories[ann['category_id']]['name'], p1, 1, 2, (255, 0, 0), 2)
            except:
                print(ann)

            cv2.rectangle(image, p1, p2, (255, 0, 0), 2)

        height_ratio = image.shape[0] / args.limit_imshow_size[0]
        width_ratio = image.shape[1] / args.limit_imshow_size[1]
        if height_ratio > 1 or width_ratio > 1:
            ratio = height_ratio if height_ratio > width_ratio else width_ratio
            new_size = int(floor(image.shape[1] / ratio)), int(floor(image.shape[0] / ratio))
            image = cv2.resize(image, new_size)

        cv2.imshow('image', image)
        if cv2.waitKey(args.delay) == 27:
            break


if __name__ == '__main__':
    main()
