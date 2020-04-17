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

""" This script removes all images and objects from annotation if category_name not in
    filter_cat.
"""

import argparse
import json


def print_stat(content):
    print('   images:', len(content['images']))
    print('   annotations:', len(content['annotations']))
    print('   categories:')
    for cat in content['categories']:
        print('      ', cat)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('input', help='Input COCO annotation (*.json).')
    args.add_argument('output', help='Output COCO annotation (*.json).')
    args.add_argument('--filter_cat', nargs='+', required=True,
                      help='Images that do not contain listed catetegories of objects will be '
                           'filtered out.')
    args.add_argument('--remap', action='store_true',
                      help='If it is true, then label map in output annotation will contain labels '
                           'from filtered categories only.')

    return args.parse_args()


def main():
    args = parse_args()

    with open(args.input) as f:
        content = json.load(f)

    print('before filtering')
    print_stat(content)

    filter_cat = args.filter_cat
    cat_id_to_name = {cat['id']: cat['name'] for cat in content['categories']}
    cat_name_to_supercategory = {cat['name']: cat['supercategory'] for cat in content['categories']}

    filtered_annotations = [ann for ann in content['annotations'] if
                            cat_id_to_name[ann['category_id']] in filter_cat]
    images_with_annotations = {ann['image_id'] for ann in filtered_annotations}

    filtered_images = [image for image in content['images'] if
                       image['id'] in images_with_annotations]

    if args.remap:
        remap = {v: (k + 1) for k, v in enumerate(args.filter_cat)}
        for ann in filtered_annotations:
            ann['category_id'] = remap[cat_id_to_name[ann['category_id']]]

        new_categories = []
        for cat_name, id in remap.items():
            new_categories.append({
                'id': id,
                'name': cat_name,
                'supercategory': cat_name_to_supercategory[cat_name]
            })
        content['categories'] = new_categories

    content['images'] = filtered_images
    content['annotations'] = filtered_annotations

    print(' ')
    print('after filtering')
    print_stat(content)

    with open(args.output, 'w') as f:
        json.dump(content, f)


if __name__ == '__main__':
    main()
