from itertools import groupby
import json
import os
import re

from dataset_converters.ConverterBase import ConverterBase


def numeric_order(string):
    return int(re.match('[0-9]+', string).group(0))


class COCO2TDGConverter(ConverterBase):

    formats = ['COCO2TDG', 'COCO2TDGSEGM']

    def __init__(self, copy_fn):
        ConverterBase.__init__(self, copy_fn)
        self.id_to_new_id = {}

    def _to_str_segm(self, instances):
        separator = '| '

        string = ' '
        for instance in instances:
            if instance['iscrowd']:
                continue
            string += str(self.id_to_new_id[instance['category_id']])
            string += ' segm '
            for component in instance['segmentation']:
                for i in component:
                    string += str(int(i)) + ' '
                string += separator
            string = string[:-len(separator)]
        return string[:-1] + '\n'

    def _to_str_bbox(self, instances, add_attr=False):
        string = ' '
        for instance in instances:
            string += str(self.id_to_new_id[instance['category_id']]) + ' '
            if add_attr:
                string += '0 '
            for i in instance['bbox']:
                string += str(int(i)) + ' '
        return string[:-1] + '\n'

    def _writelines(self, filename, lines):
        with open(filename, 'w') as f:
            f.writelines(lines)

    def _run(self, input_folder, output_folder, FORMAT):
        annotations_dir = os.path.join(input_folder, 'annotations')
        annotations_list = os.listdir(annotations_dir)

        self._ensure_folder_exists_and_is_clear(output_folder)
        for filename in annotations_list:
            image_folder = os.path.join(input_folder, filename[:-5])
            out_folder = os.path.join(output_folder, filename[:-5])
            self._ensure_folder_exists_and_is_clear(out_folder)
            annotation_file = os.path.join(annotations_dir, filename)
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

            lines_seg = []
            lines_bbox = []
            lines_bbox_attrs = []
            lines_src = []

            empty = lambda x: '\n'
            to_str_segmentation = self._to_str_segm if FORMAT == 'COCO2TDGSEGM' else empty
            to_str_bbox = self._to_str_bbox if FORMAT == 'COCO2TDG' else empty
            to_str_bbox_attr = (lambda i: self._to_str_bbox(i, add_attr=True)) if FORMAT == 'COCO2TDG' else empty

            for new_id, category in enumerate(annotations['categories'], 1):
                self.id_to_new_id[category['id']] = new_id

            c = 1
            annotations['images'].sort(key=lambda x: x['id'])
            for image in annotations['images']:
                instances = [i for i in annotations['annotations'] if i['image_id'] == image['id']]
                image_filename = image['file_name']
                ext = os.path.splitext(image_filename)[-1]
                new_filename = '{}{}'.format(c, ext)
                line_seg = new_filename
                line_bbox = new_filename
                line_bbox_attrs = new_filename

                line_seg += to_str_segmentation(instances)
                line_bbox += to_str_bbox(instances)
                line_bbox_attrs += to_str_bbox_attr(instances)

                lines_seg.append(line_seg)
                lines_bbox.append(line_bbox)
                lines_bbox_attrs.append(line_bbox_attrs)
                lines_src.append('{} {}\n'.format(new_filename, image_filename))
                self.copy(os.path.join(image_folder, image_filename), os.path.join(out_folder, new_filename))

                c += 1

            lines_seg.sort(key=numeric_order)
            lines_bbox.sort(key=numeric_order)
            lines_bbox_attrs.sort(key=numeric_order)
            lines_src.sort(key=numeric_order)

            lines_labels = ['{} {}\n'.format(self.id_to_new_id[l['id']], l['name']) for l in annotations['categories']]
            lines_labels.sort(key=numeric_order)

            self._writelines(os.path.join(out_folder, 'segms.txt'), lines_seg)
            self._writelines(os.path.join(out_folder, 'bboxes.txt'), lines_bbox)
            self._writelines(os.path.join(out_folder, 'bboxes_attr.txt'), lines_bbox_attrs)
            self._writelines(os.path.join(out_folder, 'src.txt'), lines_bbox)
            self._writelines(os.path.join(out_folder, 'labels.txt'), lines_labels)

            empty_attrs_line = '0 None\n'
            self._writelines(os.path.join(out_folder, 'attrs.txt'), [empty_attrs_line])
