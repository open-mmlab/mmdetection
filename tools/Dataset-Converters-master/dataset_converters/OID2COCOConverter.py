from dataset_converters.ConverterBase import ConverterBase

from itertools import groupby

import csv
import json
import os

import cv2


class OID2COCOConverter(ConverterBase):

    formats = ['OID2COCO']

    def __init__(self, copy_fn):
        ConverterBase.__init__(self, copy_fn)
        self.raw_name_to_label_name = {}

    def _read_label_names(self, filename):
        self.raw_name_to_label_name = {}
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for line in reader:
                self.raw_name_to_label_name[line[0]] = line[1]

    def _read_category_hierarchy(self, filename):
        self.category_hierarchy = {}
        with open(filename, 'r') as f:
            json_hierarchy = json.load(f)
        queue = []
        queue.append(json_hierarchy)
        while len(queue) > 0:
            hierarchy = queue.pop(0)
            supercategory_name = self.raw_name_to_label_name[hierarchy['LabelName']] \
                if hierarchy['LabelName'] in self.raw_name_to_label_name else 'none'
            for label in hierarchy.get('Subcategory', []):
                self.category_hierarchy[self.raw_name_to_label_name[label['LabelName']]] = supercategory_name
                queue.append(label)

    def _process_folder(self, input_folder, input_annotations, output_images_folder, output_annotations_file):
        self._ensure_folder_exists_and_is_clear(output_images_folder)

        to_dump = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}

        image_id = 1
        instance_counter = 1

        with open(input_annotations, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader, None)
            for key, group in groupby(reader, lambda x: x[0]):
                filename = key + '.jpg'
                h, w, _ = cv2.imread(os.path.join(input_folder, filename)).shape
                to_dump['images'].append(
                    {
                        'file_name': filename,
                        'height': h,
                        'width': w,
                        'id': image_id
                    }
                )
                self.copy(os.path.join(input_folder, filename), output_images_folder)

                for instance in group:
                    label = list(self.raw_name_to_label_name.values()).index(
                        self.raw_name_to_label_name[instance[2]]) + 1
                    xmin = float(instance[4]) * w
                    xmax = float(instance[5]) * w
                    ymin = float(instance[6]) * h
                    ymax = float(instance[7]) * h
                    to_dump['annotations'].append(
                        {
                            'segmentation': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                            'area': (xmax - xmin) * (ymax - ymin),
                            'iscrowd': int(instance[10] == '1'),
                            'image_id': image_id,
                            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                            'category_id': label,
                            'id': instance_counter,
                            'ignore': 0
                        }
                    )
                    instance_counter += 1
                image_id += 1

                if image_id % 1000 == 0:
                    print('Processed {} images'.format(image_id))

        for label_name, i in zip(self.raw_name_to_label_name.values(), range(len(self.raw_name_to_label_name))):
            to_dump['categories'].append(
                {
                    'supercategory': self.category_hierarchy[label_name],
                    'id': i + 1,
                    'name': label_name
                }
            )

        with open(output_annotations_file, 'w') as f:
            json.dump(to_dump, f)

    def _run_subset(self, input_folder, output_folder, subset_name):
        print('Processing subset {}'.format(subset_name))
        self._process_folder(
            os.path.join(input_folder, subset_name),
            os.path.join(input_folder, subset_name + '-annotations-bbox.csv'),
            os.path.join(output_folder, subset_name),
            os.path.join(output_folder, 'annotations', subset_name + '.json')
        )

    def _run(self, input_folder, output_folder, FORMAT):
        self._read_label_names(os.path.join(input_folder, 'class-descriptions-boxable.csv'))
        self._read_category_hierarchy(os.path.join(input_folder, 'bbox_labels_600_hierarchy.json'))

        self._ensure_folder_exists_and_is_clear(output_folder)
        self._ensure_folder_exists_and_is_clear(os.path.join(output_folder, 'annotations'))

        self._run_subset(input_folder, output_folder, 'train')
        self._run_subset(input_folder, output_folder, 'test')
        self._run_subset(input_folder, output_folder, 'validation')
