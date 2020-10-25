from dataset_converters.ConverterBase import ConverterBase

import json
import os
import re
import time

import cv2
import numpy as np

background_colors = [0, 192 * 256 * 256 + 224 * 256 + 224]

label_names = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]

class_colors = [
    (0, 0, 0),
    (0, 0, 128),
    (0, 128, 0),
    (0, 128, 128),
    (128, 0, 0),
    (128, 0, 128),
    (128, 128, 0),
    (128, 128, 128),
    (0, 0, 64),
    (0, 0, 192),
    (0, 128, 64),
    (0, 128, 192),
    (128, 0, 64),
    (128, 0, 192),
    (128, 128, 64),
    (128, 128, 192),
    (0, 64, 0),
    (0, 64, 128),
    (0, 192, 0),
    (0, 192, 128),
    (128, 64, 0),
    (192, 224, 224)
]


class VOCSEGM2COCOConverter(ConverterBase):

    formats = ['VOCSEGM2COCO']

    def __init__(self, copy_fn):
        ConverterBase.__init__(self, copy_fn)

    def _read_filenames(self, file):
        with open(file, 'r') as f:
            return [x.replace('\n', '') for x in f.readlines()]

    def _get_hw(self, image_filename):
        return cv2.imread(image_filename).shape[:2]

    def _read_segmentation(self, is_filename, cs_filename):
        instance_seg_image = cv2.imread(is_filename)
        class_seg_image = cv2.imread(cs_filename)

        b, g, r = cv2.split(instance_seg_image)
        encoded_colors = b.astype(np.int32) * 256 * 256 + g.astype(np.int32) * 256 + r.astype(np.int32)
        colors = np.unique(encoded_colors.ravel())
        for background_color in background_colors:
            colors = colors[colors != background_color]

        segms = []
        areas = []
        labels = []
        for color in colors:
            mask = np.zeros(instance_seg_image.shape[0:2], dtype=np.uint8)
            mask[encoded_colors == color] = 1
            area = np.sum(mask == 1)
            areas.append(int(area))
            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [contour for contour in contours if contour.shape[0] > 2]
            segms.append(contours)
            class_color = class_seg_image[np.nonzero(mask)][0]
            labels.append(class_colors.index(tuple(class_color)))

        return segms, areas, labels

    def _get_bbox(self, segm):
        x_min = float("inf")
        y_min = float("inf")
        x_max = 0
        y_max = 0
        for s in segm:
            x, y, w, h = cv2.boundingRect(s)
            x_b = x + w
            y_b = y + h
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x_b)
            y_max = max(y_max, y_b)

        return x_min, y_min, x_max - x_min, y_max - y_min

    def _process_images(self, names, images_folder, is_folder, cs_folder, output_images_folder, output_anno_file):
        self._ensure_folder_exists_and_is_clear(output_images_folder)

        to_dump = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}

        instance_counter = 1
        for i, name in enumerate(names):
            image_name = name + '.jpg'
            full_image_name = os.path.join(images_folder, image_name)
            seg_name = name + '.png'
            h, w = self._get_hw(full_image_name)
            segms, areas, labels = self._read_segmentation(
                os.path.join(is_folder, seg_name),
                os.path.join(cs_folder, seg_name)
            )

            to_dump['images'].append(
                {
                    'file_name': image_name,
                    'height': h,
                    'width': w,
                    'id': i + 1
                }
            )

            for seg, area, label in zip(segms, areas, labels):
                if len(seg) == 0:
                    continue
                bbox = self._get_bbox(seg)
                seg = [s.ravel().tolist() for s in seg]

                to_dump['annotations'].append(
                    {
                        'segmentation': seg,
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

        for i, label_name in enumerate(label_names):
            if i == 0:
                continue
            to_dump['categories'].append({'supercategory': 'none', 'id': i, 'name': label_name})

        with open(output_anno_file, 'w') as f:
            json.dump(to_dump, f)

    def _run(self, input_folder, output_folder, FORMAT):
        images_folder = os.path.join(input_folder, 'JPEGImages')
        instance_segmentation_folder = os.path.join(input_folder, 'SegmentationObject')
        class_segmentation_folder = os.path.join(input_folder, 'SegmentationClass')
        annotations_folder = os.path.join(output_folder, 'annotations')

        self._ensure_folder_exists_and_is_clear(output_folder)
        self._ensure_folder_exists_and_is_clear(annotations_folder)

        train_images = self._read_filenames(os.path.join(input_folder, 'ImageSets', 'Segmentation', 'train.txt'))
        val_images = self._read_filenames(os.path.join(input_folder, 'ImageSets', 'Segmentation', 'val.txt'))

        self._process_images(
            train_images,
            images_folder,
            instance_segmentation_folder,
            class_segmentation_folder,
            os.path.join(output_folder, 'train'),
            os.path.join(annotations_folder, 'train.json')
        )
        self._process_images(
            val_images,
            images_folder,
            instance_segmentation_folder,
            class_segmentation_folder,
            os.path.join(output_folder, 'val'),
            os.path.join(annotations_folder, 'val.json')
        )
