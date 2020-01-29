import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np

from .custom import CustomDataset
from .registry import DATASETS
from ..utils.utils import get_files


@DATASETS.register_module
class TobaccoDataset(CustomDataset):
    CLASSES = ('signature', 'logo')

    def __init__(self, min_size=None, **kwargs):
        super(TobaccoDataset, self).__init__(**kwargs)
        self.min_size = min_size
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}

    def parse_xml(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.iter()
        infos = [elem.attrib for elem in root]
        bboxs = []
        labels = []
        for info in infos:
            if "gedi_type" in info.keys():
                if info["gedi_type"] == "DL_PAGE":
                    width = int(info["width"])  # page width
                    height = int(info["height"])  # page height
                else:
                    labels.append(info["gedi_type"])
                    bboxs.append([int(info["col"]),   # width
                                  int(info["row"]),  # height
                                  (int(info["col"]) + int(info["width"])),
                                  (int(info["row"]) + int(info["height"]))])

        # if label is empty
        if not labels:
            labels.append([])
            bboxs.append([])
        return dict(width=width, height=height, bboxs=bboxs, labels=labels)

    def load_annotations(self, ann_folder):
        img_infos = []
        anns = get_files(ann_folder, ".xml")
        selected_imgs = get_files(self.img_prefix, ".tif")
        for ann in anns:
            filename = '{}.tif'.format(ann.split(".")[0])
            if filename in selected_imgs:
                id = ann.split(".")[0]
                xml_path = osp.join(ann_folder, ann)
                anno_info = self.parse_xml(xml_path)
                img_infos.append(
                    dict(id=id, filename=filename, width=anno_info["width"], height=anno_info["height"],
                         bboxs=anno_info["bboxs"], labels=anno_info["labels"]))
        return img_infos

    def get_ann_info(self, idx):
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        xml_dict = self.img_infos[idx]
        for bbox, label in zip(xml_dict["bboxs"], xml_dict["labels"]):
            if not label:
                break
            label = self.cat2label[label]
            difficult = False  # hardcode to False
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
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
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
