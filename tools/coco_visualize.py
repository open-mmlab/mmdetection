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

""" This scrip visualizes bounding boxes from COCO annotation. """

import argparse
import json
import os
from collections import defaultdict
import colorsys
from math import floor
import random

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from sty import bg
from tqdm import tqdm


def generate_distinct_colors(n):
    def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    def min_distance(colors_set, color_candidate):
        distances = [dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    candidates_num = 100
    hsv_colors = [(1.0, 1.0, 1.0)]
    for i in range(1, n):
        colors_candidates = [(random.random(), random.uniform(0.8, 1.0), random.uniform(0.5, 1.0)) 
                             for _ in range(candidates_num)]
        min_distances = [min_distance(hsv_colors, c) for c in colors_candidates]
        arg_max = np.argmax(min_distances)
        hsv_colors.append(colors_candidates[arg_max])

    palette = [hsv2rgb(*hsv) for hsv in hsv_colors]
    return palette


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('annotation', help='Path to COCO annotation (*.json).')
    args.add_argument('root', help='Path to images root folder.')
    args.add_argument('--delay', type=int, default=0, help='imshow delay.')
    args.add_argument('--limit_imshow_size', type=int, nargs=2, default=(1080, 1920),
                      help='Resize images with size greater than specified here size (h, w)! '
                           'keeping aspect ratio.')
    args.add_argument('--shuffle', action='store_true',
                      help='Shuffle annotation before visualization.')
    args.add_argument('--with_masks', action='store_true', help='Visualize masks as well.')

    return args.parse_args()


def print_stat(content, palette, cat_id_to_color_id):
    print('   images:', len(content['images']))
    print('   annotations:', len(content['annotations']))
    print('   categories:', len(content['categories']))
    for i, cat in enumerate(content['categories']):
        color = palette[cat_id_to_color_id[cat['id']]]
        print('      ', bg(*color) + '   ' + bg.rs + ' ' + str(cat))


def parse_segmentation(segmentation, img_h, img_w):
    if isinstance(segmentation, list):
        segmentation = [np.array([int(p) for p in s]).reshape((-1, 2)) for s in segmentation]
        mask = np.zeros((img_h, img_w), np.uint8)
        cv2.drawContours(mask, segmentation, -1, 1, -1)
    else:
        if isinstance(segmentation['counts'], list):
            rle = mask_utils.frPyObjects(segmentation, img_h, img_w)
        else:
            rle = segmentation
        mask = mask_utils.decode(rle).astype(np.uint8)
    return mask


def overlay_mask(image, masks, mask_colors):
    segments_image = image.copy()
    aggregated_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    aggregated_colored_mask = np.zeros(image.shape, dtype=np.uint8)
    black = np.zeros(3, dtype=np.uint8)
    for i, mask in enumerate(masks):
        mask = mask.astype(np.uint8)

        cv2.bitwise_or(aggregated_mask, mask, dst=aggregated_mask)
        cv2.bitwise_or(aggregated_colored_mask, np.asarray(mask_colors[i], dtype=np.uint8),
                       dst=aggregated_colored_mask, mask=mask)

    # Fill the area occupied by all instances with a colored instances mask image.
    cv2.bitwise_and(segments_image, black, dst=segments_image, mask=aggregated_mask)
    cv2.bitwise_or(segments_image, aggregated_colored_mask, dst=segments_image,
                   mask=aggregated_mask)
    # Blend original image with the one, where instances are colored.
    # As a result instances masks become transparent.
    cv2.addWeighted(image, 0.5, segments_image, 0.5, 0, dst=image)

    return image


def main():
    random.seed(0)
    args = parse_args()

    with open(args.annotation) as f:
        content = json.load(f)

    annotations = defaultdict(list)
    categories = {}

    for ann in content['annotations']:
        annotations[ann['image_id']].append(ann)

    for cat in content['categories']:
        categories[cat['id']] = cat

    cat_id_to_color_id = {cat['id']: i for i, cat in enumerate(content['categories'])}

    palette = generate_distinct_colors(len(categories))

    print_stat(content, palette, cat_id_to_color_id)

    if args.shuffle:
        random.shuffle(content['images'])

    for image_info in tqdm(content['images']):
        path = os.path.join(args.root, image_info['file_name'])
        image = cv2.imread(path)
        if image is None:
            print(path)

        masks = []
        masks_colors = []

        for ann in annotations[image_info['id']]:
            bbox = ann['bbox']
            p1 = int(bbox[0]), int(bbox[1])
            p2 = int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])

            bgr_color = palette[cat_id_to_color_id[ann['category_id']]][::-1]

            try:
                cv2.putText(image, categories[ann['category_id']]['name'], p1, 1, bbox[2] * 0.01, bgr_color, 2)
            except:
                print(ann)

            cv2.rectangle(image, p1, p2, bgr_color, 2)

            if ann['segmentation'] and args.with_masks:
                masks.append(parse_segmentation(ann['segmentation'], image.shape[0], image.shape[1]))
                masks_colors.append(bgr_color)

        image = overlay_mask(image, masks, masks_colors)

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
