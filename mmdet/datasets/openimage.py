import csv
import os.path as osp
from collections import defaultdict

import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


def list_from_csv(filename):
    item_list = defaultdict(list)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        i = -1
        for line in reader:
            i += 1
            if i == 0:
                continue
            # else:
            elif i < 10:  # for debug
                img_id = line[0]
                label = line[2]
                x_min = line[4]
                x_max = line[5]
                y_mix = line[6]
                y_max = line[7]
                IsOccluded = True if line[8] == 1 else False
                IsTruncated = True if line[9] == 1 else False
                IsGroupOf = True if line[10] == 1 else False
                IsDepiction = True if line[11] == 1 else False
                IsInside = True if line[12] == 1 else False
                # ann[img_id].append()
                item_list[img_id].append(
                    dict(
                        bbox=[x_min, x_max, y_mix, y_max],
                        label=label,
                        IsOccluded=IsOccluded,
                        IsTruncated=IsTruncated,
                        IsGroupOf=IsGroupOf,
                        IsDepiction=IsDepiction,
                        IsInside=IsInside))
    return item_list


@DATASETS.register_module()
class OpenImagesDataset(CustomDataset):

    def __init__(self, label_csv_path='', min_size=None, **kwargs):
        super(OpenImagesDataset, self).__init__(**kwargs)
        self.min_size = min_size
        self.CLASSES = self.get_classes_from_csv(label_csv_path)
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        # for debug
        self.ann = self.get_ann_info(1)

    def get_classes_from_csv(self, label_csv_path):
        """class-descriptions-boxable.csv."""
        self.label_dict = defaultdict(str)
        label_list = []
        with open(label_csv_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                self.label_dict[line[0]] = line[1]
                label_list.append(line[1])
        return label_list

    def load_annotations(self, ann_file):
        """Load annotation from csv style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            dict[list[dict]]: Annotation info from csv file.
        """

        data_infos = []
        self.ann_infos = list_from_csv(ann_file)
        img_ids = list(self.ann_infos.keys())
        for img_id in img_ids:
            filename = f'image/{img_id}.jpg'
            img_path = osp.join(self.img_prefix, filename)
            img = Image.open(img_path)
            width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))

        return data_infos

    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]['id']
        width = self.data_infos[idx]['width']
        height = self.data_infos[idx]['height']
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        IsOccludeds = []
        IsTruncateds = []
        IsGroupOfs = []
        IsDepictions = []
        IsInsides = []
        for obj in self.ann_infos[img_id]:
            name = self.label_dict[obj['label']]
            label = self.cat2label[name]
            bbox = [
                int(float(obj['bbox'][0]) * width),
                int(float(obj['bbox'][1]) * width),
                int(float(obj['bbox'][2]) * height),
                int(float(obj['bbox'][3]) * height)
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

            # Other parameters
            IsOccludeds.append(obj['IsOccluded'])
            IsTruncateds.append(obj['IsTruncated'])
            IsGroupOfs.append(obj['IsGroupOf'])
            IsDepictions.append(obj['IsDepiction'])
            IsInsides.append(obj['IsInside'])
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            # TODO: Understand why uses
            # `np.array(bboxes, ndmin=2) - 1`
            # in VOC(xml) dataset?
            bboxes = np.array(bboxes)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore)
            labels_ignore = np.array(labels_ignore)
            print()
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
