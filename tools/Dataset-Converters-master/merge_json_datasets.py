#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Merges multiple datasets in COCO format."""

import numpy as np

import argparse
import copy
from itertools import groupby
import json
import os
import shutil
import sys

from dataset_converters.utils import ensure_folder_exists_and_is_clear


def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def filter_annotations(annotations, ids):
    annotations['annotations'] = [x for x in annotations['annotations'] if x['category_id'] in ids]
    filtered_image_ids = unique([x['image_id'] for x in annotations['annotations']])

    annotations['images'] = [x for x in annotations['images'] if x['id'] in filtered_image_ids]
    annotations['categories'] = [x for x in annotations['categories'] if x['id'] in ids]


def merge_categories(merged_categories, categories):
    ids = [x['id'] for x in merged_categories]
    merged_categories.extend([x for x in categories if x['id'] not in ids])


def map_ids(annotations, ids, output_ids):
    id_to_output_id = {x: y for x, y in zip(ids, output_ids)}
    for instance in annotations['annotations']:
        instance['category_id'] = id_to_output_id[instance['category_id']]
    for category in annotations['categories']:
        category['id'] = id_to_output_id[category['id']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Typical usage: merge_json_datasets.py ' +
               '-d <path_to_images_1> -a <path_to_annotations_1> -i <list_of_ids_to_merge_1> ... ' +
               '-d <path_to_images_n> -a <path_to_annotations_n> -i <list_of_ids_to_merge_n> ' +
               '--output-ids <list_of_output_ids> -o <path_to_output_dataset>\n'
    )
    parser.add_argument('-d', '--dataset', help='path to image directories', action='append', required=True)
    parser.add_argument('-a', '--annotations', help='path to annotations', action='append', required=True)
    parser.add_argument('-o', '--output', help='path to output directory', required=True)
    parser.add_argument('-i', '--ids', help='class ids to merge', action='append', nargs='+', required=True, type=int)
    parser.add_argument('-n', '--names', help='name of classes', nargs='+')
    parser.add_argument('--output-ids', help='output label ids', nargs='+', required=True, type=int)
    args = parser.parse_args()

    ids = np.array(args.ids)
    assert(ids.shape[0] == len(args.dataset))
    assert(ids.shape[1] == len(args.output_ids))
    assert(len(args.annotations) == len(args.dataset))

    ensure_folder_exists_and_is_clear(args.output)
    ensure_folder_exists_and_is_clear(os.path.join(args.output, 'annotations'))
    ensure_folder_exists_and_is_clear(os.path.join(args.output, 'train'))

    current_image_id = 1
    current_instance_id = 1
    merged_annotations = {'annotations': [], 'categories': [], 'images': []}
    for i, (images_path, annotations_path, ids_to_merge) in enumerate(zip(args.dataset, args.annotations, ids)):
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        images_ids_with_annotations = set(unique([x['image_id'] for x in annotations['annotations']]))
        all_image_ids = {x['id'] for x in annotations['images']}
        for image_id in all_image_ids - images_ids_with_annotations:
            image_description = copy.deepcopy([x for x in annotations['images'] if x['id'] == image_id][0])
            image_description['id'] = current_image_id
            new_filename = '{}_{}'.format(i, image_description['file_name'])
            shutil.copy(
                os.path.join(images_path, image_description['file_name']),
                os.path.join(args.output, 'train', new_filename)
            )
            image_description['file_name'] = new_filename
            merged_annotations['images'].append(image_description)

            current_image_id += 1

        filter_annotations(annotations, ids_to_merge)
        map_ids(annotations, ids_to_merge, args.output_ids)
        merge_categories(merged_annotations['categories'], annotations['categories'])
        image_id_order = lambda x: x['image_id']
        annotations['annotations'].sort(key=image_id_order)
        for key, g in groupby(annotations['annotations'], image_id_order):
            image_description = copy.deepcopy([x for x in annotations['images'] if x['id'] == key][0])
            image_description['id'] = current_image_id
            for instance in g:
                instance['image_id'] = current_image_id
                instance['id'] = current_instance_id
                merged_annotations['annotations'].append(instance)

                current_instance_id += 1
            new_filename = '{}_{}'.format(i, image_description['file_name'])
            shutil.copy(
                os.path.join(images_path, image_description['file_name']),
                os.path.join(args.output, 'train', new_filename)
            )
            image_description['file_name'] = new_filename
            merged_annotations['images'].append(image_description)

            current_image_id += 1

    if (args.names is not None):
        assert(len(args.names) == len(args.output_ids))
        id_to_name = {x: y for x, y in zip(args.output_ids, args.names)}
        for category in merged_annotations['categories']:
            category['name'] = id_to_name[category['id']]

    with open(os.path.join(args.output, 'annotations', 'train.json'), 'w') as f:
        json.dump(merged_annotations, f)
