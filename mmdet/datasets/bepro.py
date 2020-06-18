# import os.path as osp
# import xml.etree.ElementTree as ET

# import mmcv
# import numpy as np

# from .custom import CustomDataset
# from .registry import DATASETS

import os.path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose

import xml.etree.ElementTree as ET
import glob
import os

@DATASETS.register_module
class BeproDataset(CustomDataset):

    def __init__(self, min_size=None, **kwargs):
        super(BeproDataset, self).__init__(**kwargs)
        self.cat2label = {'person': 1, 'ball': 2}
#        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size

    def load_annotations(self, ann_file):
        img_infos = []
        with open(ann_file, "r") as f:
            for directory_path in f:
                directory_path = directory_path[:-1]
                image_dir = os.path.join(directory_path, "img/")
                xml_dir = os.path.join(directory_path, "xml/")

                for num, filename in enumerate(sorted(glob.glob(os.path.join(image_dir,'*.jpg')))):
                    path_lst = filename.split('/')
                    img_name_lst = path_lst[-1].split('.')
                    img_id = img_name_lst[0]
                    xml_path = osp.join(xml_dir, '{}.xml'.format(img_id))
                    
                    if os.path.isfile(xml_path):
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        size = root.find('size')
                        width = int(size.find('width').text)
                        height = int(size.find('height').text)
                        img_infos.append(dict(id=img_id, filename=filename, width=width, height=height))
                    else:
                        print('[DEBUG] xml missing: ' + xml_path)
                        continue

        print(len(img_infos))
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        img_path = self.img_infos[idx]['filename']
        img_path_lst = img_path.split('/')
        img_path_lst = img_path_lst[:len(img_path_lst)-2]
        dir_path = '/'.join(img_path_lst)
        xml_path = osp.join(dir_path, 'xml', '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            
            if name not in self.cat2label:
                print('[DEBUG] invalid label: ' + xml_path + name)
                continue

            label = self.cat2label[name]
            # difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            # if difficult or ignore:
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)

        #print(len(bboxes), img_path)
        if len(bboxes) >= 37:
            print('[DEBUG] too many boxes, xml ignored! ' + img_path)
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
