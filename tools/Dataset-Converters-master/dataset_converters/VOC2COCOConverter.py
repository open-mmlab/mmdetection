from dataset_converters.ConverterBase import ConverterBase

import json
import os
import xmltodict

import cv2

# labels = [
#     'background',
#     'aeroplane',
#     'bicycle',
#     'bird',
#     'boat',
#     'bottle',
#     'bus',
#     'car',
#     'cat',
#     'chair',
#     'cow',
#     'diningtable',
#     'dog',
#     'horse',
#     'motorbike',
#     'person',
#     'pottedplant',
#     'sheep',
#     'sofa',
#     'train',
#     'tvmonitor',
# ]

labels=[
    'background',
    'water_bottle',
    'carton',
    'milk_bottle',
    'can',
]


class VOC2COCOConverter(ConverterBase):

    formats = ['VOC2COCO']

    def __init__(self, copy_fn):
        ConverterBase.__init__(self, copy_fn)

    def _get_hw(self, image_path):
        return cv2.imread(image_path).shape[:2]

    def _get_segm(self, bbox):
        x, y, w, h = bbox
        segm = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        return segm

    def _read_filenames(self, file):
        with open(file, 'r') as f:
            return [x.replace('\n', '') for x in f.readlines()]

    def _read_bbox(self, name, input_folder):

        annotations_folder = os.path.join(input_folder, 'Annotations')
        annotation_file = os.path.join(annotations_folder, name + '.xml')

        anno = {}
        with open(annotation_file, 'rb') as f:
            anno = xmltodict.parse(f)

        bboxes = []
        areas = []
        lbls = []

        objects = anno['annotation']['object']

        if not isinstance(objects, list):
            objects = [objects]

        for object in objects:

            name = object['name']
            lbls.append(labels.index(name))

            bbox_dict = object['bndbox']

            xmin = int(bbox_dict['xmin'])
            ymin = int(bbox_dict['ymin'])
            xmax = int(bbox_dict['xmax'])
            ymax = int(bbox_dict['ymax'])

            width, height = xmax - xmin, ymax - ymin

            bbox = xmin, ymin, width, height

            bboxes.append(bbox)
            areas.append(width * height)

        return bboxes, areas, lbls

    def _process_images(self, names, input_folder, images_folder, output_images_folder, output_annotations_file):

        self._ensure_folder_exists_and_is_clear(output_images_folder)

        to_dump = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}

        instance_counter = 1
        for i, name in enumerate(names):

            image_name = name + '.jpg'
            full_image_name = os.path.join(images_folder, image_name)

            h, w = self._get_hw(full_image_name)

            to_dump['images'].append(
                {
                    'file_name': image_name,
                    'height': h,
                    'width': w,
                    'id': i + 1
                }
            )

            bboxes, areas, lbls = self._read_bbox(name, input_folder)

            for bbox, area, label in zip(bboxes, areas, lbls):

                segm = self._get_segm(bbox)

                to_dump['annotations'].append(
                    {
                        'segmentation': segm,
                        'area': area,
                        'iscrowd': 0,
                        'image_id': i + 1,
                        'bbox': bbox,
                        'category_id': label,
                        'id': instance_counter,
                        'ignore': 0
                    }
                )

                instance_counter += 1

            self.copy(full_image_name, output_images_folder)

        for i, label in enumerate(labels):
            if i == 0:
                continue
            to_dump['categories'].append({'supercategory': 'none', 'id': i, 'name': label})

        with open(output_annotations_file, 'w') as f:
            json.dump(to_dump, f)

    def _run(self, input_folder, output_folder, FORMAT):

        images_folder = os.path.join(input_folder, 'JPEGImages')

        train_images = self._read_filenames(os.path.join(input_folder, 'ImageSets', 'Main', 'train.txt'))
        val_images = self._read_filenames(os.path.join(input_folder, 'ImageSets', 'Main', 'val.txt'))

        annotations_folder = os.path.join(output_folder, 'Annotations')

        self._ensure_folder_exists_and_is_clear(output_folder)
        self._ensure_folder_exists_and_is_clear(annotations_folder)

        self._process_images(val_images, input_folder, images_folder, os.path.join(output_folder, 'val'),
            os.path.join(annotations_folder, 'val.json'))
        self._process_images(train_images, input_folder, images_folder, os.path.join(output_folder, 'train'),
            os.path.join(annotations_folder, 'train.json'))
