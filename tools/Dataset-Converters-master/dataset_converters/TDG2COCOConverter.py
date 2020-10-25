#!/usr/bin/env python2

from itertools import groupby
import json
import os

import cv2

from dataset_converters.ConverterBase import ConverterBase


class TDG2COCOConverter(ConverterBase):

    formats = ['TDGSEGM2COCO', 'TDG2COCO']

    def __init__(self, copy_fn):
        ConverterBase.__init__(self, copy_fn)

    def _read_annotations(self, filename):
        with open(filename, 'r') as f:
            lines = [l.strip().split(' ') for l in f.readlines()]
        instances = {}

        def group(lst, n):
            return zip(*[lst[i::n] for i in range(n)])

        for line in lines:
            filename = line[0]
            instances[filename] = []
            for instance in group(line[1:], 5):
                bbox = [int(s) for s in instance[1:]]
                x, y, w, h = bbox
                segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                instances[filename].append({'id': int(instance[0]), 'bbox': bbox, 'segmentation': segmentation})

        return instances

    def _read_annotations_with_segmentations(self, filename):
        with open(filename, 'r') as f:
            lines = [l.strip().split(' ') for l in f.readlines()]
        instances = {}
        for line in lines:
            filename = line[0]
            instances[filename] = []
            label = int(line[1])
            next_label = -1
            for instance in [list(group) for k, group in groupby(line[2:], lambda x: x == 'segm') if not k]:
                assert(label >= 0)
                # If the number of elements is odd there is a label for next instance
                instance = [int(s) for s in instance]
                if (len(instance) % 2 == 1):
                    next_label = instance[-1]
                    instance = instance[:-1]
                min_x = min(instance[::2])
                max_x = max(instance[::2])
                min_y = min(instance[1::2])
                max_y = max(instance[1::2])
                bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
                instances[filename].append({'id': label, 'bbox': bbox, 'segmentation': [instance]})
                label = next_label

        return instances

    def _read_labels(self, filename):
        with open(filename, 'r') as f:
            lines = [l.strip().split(' ') for l in f.readlines()]
        labels = []
        for line in lines:
            labels.append({'supercategory': 'none', 'id': int(line[0]), 'name': line[1]})

        return labels

    def _run(self, input_folder, output_folder, FORMAT):
        images_folder = os.path.join(output_folder, 'train')
        annotations_folder = os.path.join(output_folder, 'annotations')

        self._ensure_folder_exists_and_is_clear(output_folder)
        self._ensure_folder_exists_and_is_clear(images_folder)
        self._ensure_folder_exists_and_is_clear(annotations_folder)

        labels_file = os.path.join(input_folder, 'labels.txt')

        if FORMAT == 'TDGSEGM2COCO':
            instances = self._read_annotations_with_segmentations(os.path.join(input_folder, 'segms.txt'))
        else:
            instances = self._read_annotations(os.path.join(input_folder, 'bboxes.txt'))
        labels = self._read_labels(labels_file)

        to_dump = {'images': [], 'type': 'instances', 'annotations': [], 'categories': labels}
        image_counter = 1
        instance_counter = 1
        for filename, single_image_instances in instances.items():
            full_image_path = os.path.join(input_folder, filename)
            image = cv2.imread(full_image_path)
            to_dump['images'].append(
                {
                    'file_name': filename,
                    'height': image.shape[0],
                    'width': image.shape[1],
                    'id': image_counter
                }
            )
            for instance in single_image_instances:
                _, _, w, h = instance['bbox']
                to_dump['annotations'].append(
                    {
                        'segmentation': instance['segmentation'],
                        'area': w * h,
                        'iscrowd': 0,
                        'image_id': image_counter,
                        'bbox': instance['bbox'],
                        'category_id': instance['id'],
                        'id': instance_counter,
                        'ignore': 0
                    }
                )
                instance_counter += 1
            self.copy(full_image_path, images_folder)
            image_counter += 1

        with open(os.path.join(annotations_folder, 'train.json'), 'w') as f:
            json.dump(to_dump, f, indent=4)
