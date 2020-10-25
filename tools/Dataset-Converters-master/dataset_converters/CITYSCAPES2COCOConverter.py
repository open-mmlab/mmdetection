import json
import os

import cv2
import numpy as np

from cityscapesscripts.helpers.labels import labels, name2label
from cityscapesscripts.evaluation.instances2dict import instances2dict
from cityscapesscripts.evaluation.instance import Instance

from dataset_converters.ConverterBase import ConverterBase


class CITYSCAPES2COCOConverter(ConverterBase):

    formats = ['CITYSCAPES2COCO']

    def __init__(self, copy_fn):

        ConverterBase.__init__(self, copy_fn)

    def _reset_dump(self):

        self.image_id = 0
        self.annotation_id = 0

        self.to_dump = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}

    def _parse_json(self, f):

        with open(f) as json_file:
            json_dict = json.load(json_file)

        f_basename = os.path.basename(f)

        image_var_name = f_basename[:f.rfind('_')]
        image_var_name = image_var_name[:image_var_name.rfind('_')]

        img_inst_name = image_var_name + '_instanceIds.png'
        img_inst_name = os.path.join(os.path.dirname(f), img_inst_name)
        self.img_inst_name = os.path.abspath(img_inst_name)

        image_var_name = image_var_name[:image_var_name.rfind('_')]
        image_name = image_var_name + '_leftImg8bit.png'

        self.image_name = image_name

        self.to_dump['images'].append(
            {
                'file_name': image_name,
                'height': json_dict['imgHeight'],
                'width': json_dict['imgWidth'],
                'id': self.image_id
            }
        )

    def _run(self, input_folder, output_folder, FORMAT):

        train_path = os.path.join(input_folder, 'gtFine_trainvaltest', 'gtFine', 'train')
        assert os.path.isdir(train_path)

        val_path = os.path.join(input_folder, 'gtFine_trainvaltest', 'gtFine', 'val')
        assert os.path.isdir(val_path)

        dataset_folders = [train_path, val_path]
        dataset_names = ['train', 'val']
        input_image_folders = [os.path.join(input_folder, 'leftImg8bit_trainvaltest', 'leftImg8bit', 'train'),
                               os.path.join(input_folder, 'leftImg8bit_trainvaltest', 'leftImg8bit', 'val')]

        self._ensure_folder_exists_and_is_clear(output_folder)

        label_names = [label.name for label in labels if label.hasInstances and not label.ignoreInEval]

        annotation_folder = os.path.join(output_folder, 'annotations')
        self._ensure_folder_exists_and_is_clear(annotation_folder)

        for input_image_folder, dataset_folder, dataset_name in zip(input_image_folders, dataset_folders, dataset_names):

            self._reset_dump()

            image_folder = os.path.join(output_folder, dataset_name)
            self._ensure_folder_exists_and_is_clear(image_folder)

            label2name = {}
            for cat_id, label_name in enumerate(label_names):
                label2name[name2label[label_name].id] = label_name
                self.to_dump['categories'].append({'id': cat_id + 1, 'name': label_name})

            for city_folder in os.listdir(dataset_folder):

                input_image_folder_city = os.path.join(input_image_folder, city_folder)

                assert os.path.isdir(os.path.join(dataset_folder, city_folder))
                for file_name in os.listdir(os.path.join(dataset_folder, city_folder)):

                    file_name = os.path.join(dataset_folder, city_folder, file_name)
                    assert os.path.isfile(file_name)

                    if file_name.endswith('_polygons.json'):
                        self._parse_json(file_name)

                        img_inst = cv2.imread(self.img_inst_name, cv2.IMREAD_ANYDEPTH)
                        instance_dict = instances2dict([self.img_inst_name], verbose=False)[self.img_inst_name]

                        for instanceId in np.unique(img_inst):

                            if instanceId < 1000:
                                continue

                            label = None
                            for value in list(instance_dict.values()):
                                for list_value in value:
                                    if list_value['instID'] == instanceId:
                                        label = list_value['labelID']
                                        break

                            assert label is not None

                            if label not in label2name:
                                continue

                            label_to_dump = None
                            for category in self.to_dump['categories']:
                                if category['name'] == label2name[label]:
                                    label_to_dump = category['id']
                                    break

                            assert label_to_dump is not None

                            mask = (img_inst == instanceId).astype(np.uint8)
                            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            x_min = float("inf")
                            y_min = float("inf")
                            x_max = 0
                            y_max = 0
                            for s in contours:
                                x, y, w, h = cv2.boundingRect(s)
                                x_b = x + w
                                y_b = y + h
                                x_min = min(x_min, x)
                                y_min = min(y_min, y)
                                x_max = max(x_max, x_b)
                                y_max = max(y_max, y_b)

                            area = int(np.sum(mask == 1))
                            segm = [contour.ravel().tolist() for contour in contours]

                            self.to_dump['annotations'].append(
                                {
                                    'segmentation': segm,
                                    'area': area,
                                    'iscrowd': 0,
                                    'image_id': self.image_id,
                                    'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                                    'category_id': label_to_dump,
                                    'id': self.annotation_id,
                                    'ignore': 0
                                }
                            )

                            self.annotation_id += 1

                        self.image_id += 1
                        self.copy(os.path.join(input_image_folder_city, self.image_name), image_folder)

            with open(os.path.join(annotation_folder, dataset_name + '.json'), 'w') as f:
                json.dump(self.to_dump, f)
