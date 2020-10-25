from dataset_converters.ConverterBase import ConverterBase

import json
import os
import re
import time

import cv2
import numpy as np

background_color = 0


class ADE20K2COCOConverter(ConverterBase):

    formats = ['ADE20K2COCO']

    def __init__(self, copy_fn):
        ConverterBase.__init__(self, copy_fn)
        self.label_names = []

    def _get_files(self, folder):
        all_files = []
        for root, dirs, files in os.walk(folder):
            all_files.extend([os.path.join(root, file) for file in files])
        return all_files

    def _get_by_pattern(self, pattern, files):
        return [file for file in files if re.match(pattern, file) is not None]

    def _get_image_filenames(self, files):
        return self._get_by_pattern('.*jpg', files)

    def _get_segmentation_filenames(self, files):
        return self._get_by_pattern('.*seg[.]png', files)

    def _get_attribute_filenames(self, files):
        return self._get_by_pattern('.*atr[.]txt', files)

    def _get_hw(self, image_filename):
        return cv2.imread(image_filename).shape[:2]

    def _read_class_names(self, attribute_filename):
        with open(attribute_filename, 'r') as f:
            lines = [line.split(' # ') for line in f.readlines()]
        class_names = [line[4] for line in lines if line[1] == '0']
        return class_names

    def _read_segmentation(self, segmentation_filename):
        seg_image = cv2.imread(segmentation_filename)

        b, g, r = cv2.split(seg_image)
        encoded_colors = b.astype(np.int32) * 256 * 256 + g.astype(np.int32) * 256 + r.astype(np.int32)
        colors = np.unique(encoded_colors.ravel())
        colors = colors[colors != background_color]

        segms = []
        areas = []
        for color in colors:
            mask = np.zeros(seg_image.shape[0:2], dtype=np.uint8)
            mask[encoded_colors == color] = 1
            area = np.sum(mask == 1)
            areas.append(int(area))
            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [contour for contour in contours if contour.shape[0] > 2]
            segms.append(contours)

        return segms, areas

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

    def _process_folder(self, intput_folder, output_images_folder, output_annotations_file):
        self._ensure_folder_exists_and_is_clear(output_images_folder)

        files = self._get_files(intput_folder)
        files.sort()

        image_filenames = self._get_image_filenames(files)
        segmentation_filenames = self._get_segmentation_filenames(files)
        attribute_filenames = self._get_attribute_filenames(files)

        images_count = len(image_filenames)
        assert(images_count == len(segmentation_filenames) and images_count == len(attribute_filenames))
        to_dump = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}

        instance_counter = 1
        iterables = zip(image_filenames, segmentation_filenames, attribute_filenames)
        for i, (image, segmentation, annotation) in enumerate(iterables, 1):
            h, w = self._get_hw(image)
            segms, areas = self._read_segmentation(segmentation)
            class_names = self._read_class_names(annotation)

            if len(segms) != len(class_names):
                print('Skipping file {}. Number of colors {} does not match the number of attributes {}.\n'.format(
                    image, len(segms), len(class_names)))
                continue

            to_dump['images'].append(
                {
                    'file_name': os.path.basename(image),
                    'height': h,
                    'width': w,
                    'id': i
                }
            )

            for seg, area, class_name in zip(segms, areas, class_names):
                if len(seg) == 0:
                    continue
                bbox = self._get_bbox(seg)
                seg = [s.ravel().tolist() for s in seg]

                if class_name in self.label_names:
                    category_id = self.label_names.index(class_name) + 1
                else:
                    self.label_names.append(class_name)
                    category_id = len(self.label_names)

                to_dump['annotations'].append(
                    {
                        'segmentation': seg,
                        'area': area,
                        'iscrowd': 0,
                        'image_id': i,
                        'bbox': bbox,
                        'category_id': category_id,
                        'id': instance_counter,
                        'ignore': 0
                    }
                )
                instance_counter += 1

            self.copy(image, output_images_folder)

        for label_name, i in zip(self.label_names, range(1, len(self.label_names) + 1)):
            to_dump['categories'].append({'supercategory': 'none', 'id': i, 'name': label_name})

        with open(output_annotations_file, 'w') as f:
            json.dump(to_dump, f)

    def _run(self, input_folder, output_folder, FORMAT):
        images_folder = os.path.join(input_folder, 'images')
        annotations_folder = os.path.join(output_folder, 'annotations')

        self._ensure_folder_exists_and_is_clear(output_folder)
        self._ensure_folder_exists_and_is_clear(annotations_folder)

        self._process_folder(
            os.path.join(images_folder, 'training'),
            os.path.join(output_folder, 'train'),
            os.path.join(annotations_folder, 'train.json')
        )
        self._process_folder(
            os.path.join(images_folder, 'validation'),
            os.path.join(output_folder, 'val'),
            os.path.join(annotations_folder, 'val.json')
        )
