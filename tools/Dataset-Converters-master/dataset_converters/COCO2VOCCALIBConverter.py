import dicttoxml
import json
import os
import re

from collections import OrderedDict
from xml.dom.minidom import parseString

import cv2

from dataset_converters.ConverterBase import ConverterBase


class COCO2VOCCALIBConverter(ConverterBase):

    formats = ['COCO2VOCCALIB']

    def __init__(self, copy_fn):
        ConverterBase.__init__(self, copy_fn)
        self.id_to_new_id = {}

    def _create_output_folder_layout(self, output_folder, json_filename):

        dir_name = os.path.join(output_folder, os.path.splitext(json_filename)[0])
        self._ensure_folder_exists_and_is_clear(dir_name)

        annotation_dir = os.path.join(dir_name, 'Annotations')
        self._ensure_folder_exists_and_is_clear(annotation_dir)

        imageSets_dir = os.path.join(dir_name, 'ImageSets')
        self._ensure_folder_exists_and_is_clear(imageSets_dir)

        jpegImages_dir = os.path.join(dir_name, 'JPEGImages')
        self._ensure_folder_exists_and_is_clear(jpegImages_dir)

        segmentationClass_dir = os.path.join(dir_name, 'SegmentationClass')
        self._ensure_folder_exists_and_is_clear(segmentationClass_dir)

        segmentationObject_dir = os.path.join(dir_name, 'SegmentationObject')
        self._ensure_folder_exists_and_is_clear(segmentationObject_dir)

        return dir_name, annotation_dir, imageSets_dir, jpegImages_dir, segmentationClass_dir, segmentationObject_dir

    def _dump_boxes(self, instances, image_filename, folder, categories, annotation_dir):

        image_filename_base = os.path.basename(image_filename)

        anno_dict = OrderedDict()
        anno_dict['folder'] = os.path.basename(os.path.normpath(folder))
        anno_dict['filename'] = image_filename_base

        database_name = os.path.basename(os.path.normpath(self.input_folder)) \
            + os.path.basename(os.path.normpath(folder))

        anno_dict['source'] = OrderedDict()
        anno_dict['source']['database'] = database_name
        anno_dict['source']['annotation'] = 'PASCAL VOC2007'
        anno_dict['source']['image'] = database_name

        image_filename_full = os.path.join(self.input_folder, anno_dict['folder'], image_filename_base)
        img = cv2.imread(image_filename_full)
        height, width, depth = img.shape

        anno_dict['size'] = OrderedDict()
        anno_dict['size']['height'] = height
        anno_dict['size']['width'] = width
        anno_dict['size']['depth'] = depth

        anno_dict['segmented'] = 1

        for idx, instance in enumerate(instances):

            bbox = []
            for i in instance['bbox']:
                bbox.append(int(i))

            new_object = OrderedDict()
            new_object['name'] = categories[instance['category_id']]
            new_object['pose'] = 'Unspecified'
            new_object['truncated'] = 0
            new_object['difficult'] = 0
            new_object['bndbox'] = OrderedDict()
            new_object['bndbox']['xmin'] = bbox[0]
            new_object['bndbox']['ymin'] = bbox[1]
            new_object['bndbox']['xmax'] = bbox[0] + bbox[2]
            new_object['bndbox']['ymax'] = bbox[1] + bbox[3]

            anno_dict['object' + str(idx)] = new_object

        xml_filename = os.path.join(annotation_dir, os.path.splitext(image_filename_base)[0] + '.xml')
        xml_content = dicttoxml.dicttoxml(anno_dict, attr_type=False)
        xml_content = parseString(xml_content)
        xml_content = xml_content.toprettyxml(indent='    ')

        xml_content = re.sub('object[0-9]+>', 'object>', xml_content)
        xml_content = re.sub('<root>', '<annotation>', xml_content)
        xml_content = re.sub('</root>', '</annotation>', xml_content)
        xml_content = re.sub('<\?xml.*\n', '', xml_content)

        with open(xml_filename, 'w') as f:
            f.write(xml_content)

    def _run(self, input_folder, output_folder, FORMAT):

        self._ensure_folder_exists_and_is_clear(output_folder)
        self.input_folder = input_folder

        annotations_dir = os.path.join(input_folder, 'annotations')
        annotations_list = os.listdir(annotations_dir)

        for filename in annotations_list:

            dir_name, annotation_dir, imageSets_dir, jpegImages_dir, segmentationClass_dir, segmentationObject_dir = \
                self._create_output_folder_layout(output_folder, filename)

            image_folder = os.path.join(input_folder, filename[:-5])

            annotation_file = os.path.join(annotations_dir, filename)
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

            new_id = 1
            for category in annotations['categories']:
                self.id_to_new_id[category['id']] = new_id
                new_id += 1

            catId2Name = {l['id']: l['name'] for l in annotations['categories']}
            labels = ['{} {}'.format(l['name'], self.id_to_new_id[l['id']]) for l in annotations['categories']]

            c = 1
            annotations['images'].sort(key=lambda x: x['id'])

            self._ensure_folder_exists_and_is_clear(os.path.join(imageSets_dir, 'Action'))
            self._ensure_folder_exists_and_is_clear(os.path.join(imageSets_dir, 'Layout'))

            main_set_path = os.path.join(imageSets_dir, 'Main')
            segmentation_set_path = os.path.join(imageSets_dir, 'Segmentation')

            self._ensure_folder_exists_and_is_clear(main_set_path)
            self._ensure_folder_exists_and_is_clear(segmentation_set_path)

            with open(os.path.join(dir_name, 'VOC_SSD_Classes.txt'), 'w') as flabel:
                flabel.write('none_of_the_above 0')
                for label_line in labels:
                    flabel.write('\n')
                    flabel.write(label_line)

            for image in annotations['images']:

                instances = [i for i in annotations['annotations'] if i['image_id'] == image['id']]

                image_filename = image['file_name']

                image_filename_base = os.path.basename(image_filename)

                self._dump_boxes(instances, image_filename, dir_name, catId2Name, annotation_dir)
                f_set_name = os.path.join(main_set_path, os.path.basename(os.path.normpath(dir_name)) + '.txt')
                with open(f_set_name, 'a') as f_set:
                    f_set.write(image_filename_base + '\n')

                self.copy(os.path.join(image_folder, image_filename_base),
                          os.path.join(jpegImages_dir, image_filename_base))
