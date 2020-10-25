from dataset_converters.ConverterBase import ConverterBase

import json
import os
import xml.etree.ElementTree as ET


class CVAT2COCOConverter(ConverterBase):

    formats = ['CVAT2COCO']

    def __init__(self, copy_fn):
        ConverterBase.__init__(self, copy_fn)

    def _run(self, input_folder, output_folder, FORMAT):
        dataset_name = 'cvat'
        xml_annotations_path = os.path.join(input_folder, 'annotations.xml')
        output_annotations_folder = os.path.join(output_folder, 'annotations')
        output_images_folder = os.path.join(output_folder, dataset_name)

        self._ensure_folder_exists_and_is_clear(output_folder)
        self._ensure_folder_exists_and_is_clear(output_annotations_folder)
        self._ensure_folder_exists_and_is_clear(output_images_folder)

        to_dump = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}

        tree = ET.parse(xml_annotations_path)
        root = tree.getroot()
        meta = next(root.iter('meta'))
        label_names = []
        for i, label in enumerate(next(meta.iter('labels'))):
            to_dump['categories'].append(
                {
                    'supercategory': 'none',
                    'id': i + 1,
                    'name': next(label.iter('name')).text

                })
            label_names.append(next(label.iter('name')).text)
        instance_counter = 0
        for image_id, image in enumerate(root.iter('image')):
            basename = os.path.basename(image.attrib['name'])
            to_dump['images'].append(
                {
                    'file_name': basename,
                    'height': int(image.attrib['height']),
                    'width': int(image.attrib['width']),
                    'id': image_id
                }
            )
            for box in image.iter('box'):
                xmin = float(box.attrib['xtl'])
                ymin = float(box.attrib['ytl'])
                xmax = float(box.attrib['xbr'])
                ymax = float(box.attrib['ybr'])
                to_dump['annotations'].append(
                    {
                        'segmentation': [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]],
                        'area': (xmax - xmin) * (ymax - ymin),
                        'iscrowd': 0,
                        'image_id': image_id,
                        'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                        'category_id': label_names.index(box.attrib['label']) + 1,
                        'id': instance_counter,
                        'ignore': 0
                    }
                )

                instance_counter += 1

            for polygon in image.iter('polygon'):

                points_attrib = polygon.attrib['points']
                points_attrib_list = points_attrib.split(';')
                segmentation = []
                points = []
                for x_y in points_attrib_list:
                    x, y = x_y.split(',')
                    points.append(float(x))
                    points.append(float(y))

                segmentation += [points]

                xmin = None
                ymin = None
                xmax = None
                ymax = None

                for component in segmentation:
                    assert len(component) % 2 == 0
                    for i in range(0, len(component), 2):

                        x, y = component[i], component[i + 1]

                        if xmin is not None:
                            xmin = min(xmin, x)
                            ymin = min(ymin, y)
                            xmax = max(xmax, x)
                            ymax = max(ymax, y)
                        else:
                            xmin = x
                            ymin = y
                            xmax = x
                            ymax = y

                to_dump['annotations'].append(
                    {
                        'segmentation': segmentation,
                        'area': (xmax - xmin) * (ymax - ymin),
                        'iscrowd': 0,
                        'image_id': image_id,
                        'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                        'category_id': label_names.index(polygon.attrib['label']) + 1,
                        'id': instance_counter,
                        'ignore': 0
                    }
                )

                instance_counter += 1

            self.copy(os.path.join(input_folder, basename), output_images_folder)

        with open(os.path.join(output_annotations_folder, dataset_name + '.json'), 'w') as f:
            json.dump(to_dump, f)
