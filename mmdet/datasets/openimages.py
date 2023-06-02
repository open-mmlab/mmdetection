# Copyright (c) OpenMMLab. All rights reserved.
import csv
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from mmengine.fileio import get_local_path, load
from mmengine.utils import is_abs

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class OpenImagesDataset(BaseDetDataset):
    """Open Images dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        label_file (str): File path of the label description file that
            maps the classes names in MID format to their short
            descriptions.
        meta_file (str): File path to get image metas.
        hierarchy_file (str): The file path of the class hierarchy.
        image_level_ann_file (str): Human-verified image level annotation,
            which is used in evaluation.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    METAINFO: dict = dict(dataset_type='oid_v6')

    def __init__(self,
                 label_file: str,
                 meta_file: str,
                 hierarchy_file: str,
                 image_level_ann_file: Optional[str] = None,
                 **kwargs) -> None:
        self.label_file = label_file
        self.meta_file = meta_file
        self.hierarchy_file = hierarchy_file
        self.image_level_ann_file = image_level_ann_file
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """
        classes_names, label_id_mapping = self._parse_label_file(
            self.label_file)
        self._metainfo['classes'] = classes_names
        self.label_id_mapping = label_id_mapping

        if self.image_level_ann_file is not None:
            img_level_anns = self._parse_img_level_ann(
                self.image_level_ann_file)
        else:
            img_level_anns = None

        # OpenImagesMetric can get the relation matrix from the dataset meta
        relation_matrix = self._get_relation_matrix(self.hierarchy_file)
        self._metainfo['RELATION_MATRIX'] = relation_matrix

        data_list = []
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            with open(local_path, 'r') as f:
                reader = csv.reader(f)
                last_img_id = None
                instances = []
                for i, line in enumerate(reader):
                    if i == 0:
                        continue
                    img_id = line[0]
                    if last_img_id is None:
                        last_img_id = img_id
                    label_id = line[2]
                    assert label_id in self.label_id_mapping
                    label = int(self.label_id_mapping[label_id])
                    bbox = [
                        float(line[4]),  # xmin
                        float(line[6]),  # ymin
                        float(line[5]),  # xmax
                        float(line[7])  # ymax
                    ]
                    is_occluded = True if int(line[8]) == 1 else False
                    is_truncated = True if int(line[9]) == 1 else False
                    is_group_of = True if int(line[10]) == 1 else False
                    is_depiction = True if int(line[11]) == 1 else False
                    is_inside = True if int(line[12]) == 1 else False

                    instance = dict(
                        bbox=bbox,
                        bbox_label=label,
                        ignore_flag=0,
                        is_occluded=is_occluded,
                        is_truncated=is_truncated,
                        is_group_of=is_group_of,
                        is_depiction=is_depiction,
                        is_inside=is_inside)
                    last_img_path = osp.join(self.data_prefix['img'],
                                             f'{last_img_id}.jpg')
                    if img_id != last_img_id:
                        # switch to a new image, record previous image's data.
                        data_info = dict(
                            img_path=last_img_path,
                            img_id=last_img_id,
                            instances=instances,
                        )
                        data_list.append(data_info)
                        instances = []
                    instances.append(instance)
                    last_img_id = img_id
                data_list.append(
                    dict(
                        img_path=last_img_path,
                        img_id=last_img_id,
                        instances=instances,
                    ))

        # add image metas to data list
        img_metas = load(
            self.meta_file, file_format='pkl', backend_args=self.backend_args)
        assert len(img_metas) == len(data_list)
        for i, meta in enumerate(img_metas):
            img_id = data_list[i]['img_id']
            assert f'{img_id}.jpg' == osp.split(meta['filename'])[-1]
            h, w = meta['ori_shape'][:2]
            data_list[i]['height'] = h
            data_list[i]['width'] = w
            # denormalize bboxes
            for j in range(len(data_list[i]['instances'])):
                data_list[i]['instances'][j]['bbox'][0] *= w
                data_list[i]['instances'][j]['bbox'][2] *= w
                data_list[i]['instances'][j]['bbox'][1] *= h
                data_list[i]['instances'][j]['bbox'][3] *= h
            # add image-level annotation
            if img_level_anns is not None:
                img_labels = []
                confidences = []
                img_ann_list = img_level_anns.get(img_id, [])
                for ann in img_ann_list:
                    img_labels.append(int(ann['image_level_label']))
                    confidences.append(float(ann['confidence']))
                data_list[i]['image_level_labels'] = np.array(
                    img_labels, dtype=np.int64)
                data_list[i]['confidences'] = np.array(
                    confidences, dtype=np.float32)
        return data_list

    def _parse_label_file(self, label_file: str) -> tuple:
        """Get classes name and index mapping from cls-label-description file.

        Args:
            label_file (str): File path of the label description file that
                maps the classes names in MID format to their short
                descriptions.

        Returns:
            tuple: Class name of OpenImages.
        """

        index_list = []
        classes_names = []
        with get_local_path(
                label_file, backend_args=self.backend_args) as local_path:
            with open(local_path, 'r') as f:
                reader = csv.reader(f)
                for line in reader:
                    # self.cat2label[line[0]] = line[1]
                    classes_names.append(line[1])
                    index_list.append(line[0])
        index_mapping = {index: i for i, index in enumerate(index_list)}
        return classes_names, index_mapping

    def _parse_img_level_ann(self,
                             img_level_ann_file: str) -> Dict[str, List[dict]]:
        """Parse image level annotations from csv style ann_file.

        Args:
            img_level_ann_file (str): CSV style image level annotation
                file path.

        Returns:
            Dict[str, List[dict]]: Annotations where item of the defaultdict
            indicates an image, each of which has (n) dicts.
            Keys of dicts are:

                - `image_level_label` (int): Label id.
                - `confidence` (float): Labels that are human-verified to be
                  present in an image have confidence = 1 (positive labels).
                  Labels that are human-verified to be absent from an image
                  have confidence = 0 (negative labels). Machine-generated
                  labels have fractional confidences, generally >= 0.5.
                  The higher the confidence, the smaller the chance for
                  the label to be a false positive.
        """

        item_lists = defaultdict(list)
        with get_local_path(
                img_level_ann_file,
                backend_args=self.backend_args) as local_path:
            with open(local_path, 'r') as f:
                reader = csv.reader(f)
                for i, line in enumerate(reader):
                    if i == 0:
                        continue
                    img_id = line[0]
                    item_lists[img_id].append(
                        dict(
                            image_level_label=int(
                                self.label_id_mapping[line[2]]),
                            confidence=float(line[3])))
        return item_lists

    def _get_relation_matrix(self, hierarchy_file: str) -> np.ndarray:
        """Get the matrix of class hierarchy from the hierarchy file. Hierarchy
        for 600 classes can be found at https://storage.googleapis.com/openimag
        es/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html.

        Args:
            hierarchy_file (str): File path to the hierarchy for classes.

        Returns:
            np.ndarray: The matrix of the corresponding relationship between
            the parent class and the child class, of shape
            (class_num, class_num).
        """  # noqa

        hierarchy = load(
            hierarchy_file, file_format='json', backend_args=self.backend_args)
        class_num = len(self._metainfo['classes'])
        relation_matrix = np.eye(class_num, class_num)
        relation_matrix = self._convert_hierarchy_tree(hierarchy,
                                                       relation_matrix)
        return relation_matrix

    def _convert_hierarchy_tree(self,
                                hierarchy_map: dict,
                                relation_matrix: np.ndarray,
                                parents: list = [],
                                get_all_parents: bool = True) -> np.ndarray:
        """Get matrix of the corresponding relationship between the parent
        class and the child class.

        Args:
            hierarchy_map (dict): Including label name and corresponding
                subcategory. Keys of dicts are:

                - `LabeName` (str): Name of the label.
                - `Subcategory` (dict | list): Corresponding subcategory(ies).
            relation_matrix (ndarray): The matrix of the corresponding
                relationship between the parent class and the child class,
                of shape (class_num, class_num).
            parents (list): Corresponding parent class.
            get_all_parents (bool): Whether get all parent names.
                Default: True

        Returns:
            ndarray: The matrix of the corresponding relationship between
            the parent class and the child class, of shape
            (class_num, class_num).
        """

        if 'Subcategory' in hierarchy_map:
            for node in hierarchy_map['Subcategory']:
                if 'LabelName' in node:
                    children_name = node['LabelName']
                    children_index = self.label_id_mapping[children_name]
                    children = [children_index]
                else:
                    continue
                if len(parents) > 0:
                    for parent_index in parents:
                        if get_all_parents:
                            children.append(parent_index)
                        relation_matrix[children_index, parent_index] = 1
                relation_matrix = self._convert_hierarchy_tree(
                    node, relation_matrix, parents=children)
        return relation_matrix

    def _join_prefix(self):
        """Join ``self.data_root`` with annotation path."""
        super()._join_prefix()
        if not is_abs(self.label_file) and self.label_file:
            self.label_file = osp.join(self.data_root, self.label_file)
        if not is_abs(self.meta_file) and self.meta_file:
            self.meta_file = osp.join(self.data_root, self.meta_file)
        if not is_abs(self.hierarchy_file) and self.hierarchy_file:
            self.hierarchy_file = osp.join(self.data_root, self.hierarchy_file)
        if self.image_level_ann_file and not is_abs(self.image_level_ann_file):
            self.image_level_ann_file = osp.join(self.data_root,
                                                 self.image_level_ann_file)


@DATASETS.register_module()
class OpenImagesChallengeDataset(OpenImagesDataset):
    """Open Images Challenge dataset for detection.

    Args:
        ann_file (str): Open Images Challenge box annotation in txt format.
    """

    METAINFO: dict = dict(dataset_type='oid_challenge')

    def __init__(self, ann_file: str, **kwargs) -> None:
        if not ann_file.endswith('txt'):
            raise TypeError('The annotation file of Open Images Challenge '
                            'should be a txt file.')

        super().__init__(ann_file=ann_file, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """
        classes_names, label_id_mapping = self._parse_label_file(
            self.label_file)
        self._metainfo['classes'] = classes_names
        self.label_id_mapping = label_id_mapping

        if self.image_level_ann_file is not None:
            img_level_anns = self._parse_img_level_ann(
                self.image_level_ann_file)
        else:
            img_level_anns = None

        # OpenImagesMetric can get the relation matrix from the dataset meta
        relation_matrix = self._get_relation_matrix(self.hierarchy_file)
        self._metainfo['RELATION_MATRIX'] = relation_matrix

        data_list = []
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            with open(local_path, 'r') as f:
                lines = f.readlines()
        i = 0
        while i < len(lines):
            instances = []
            filename = lines[i].rstrip()
            i += 2
            img_gt_size = int(lines[i])
            i += 1
            for j in range(img_gt_size):
                sp = lines[i + j].split()
                instances.append(
                    dict(
                        bbox=[
                            float(sp[1]),
                            float(sp[2]),
                            float(sp[3]),
                            float(sp[4])
                        ],
                        bbox_label=int(sp[0]) - 1,  # labels begin from 1
                        ignore_flag=0,
                        is_group_ofs=True if int(sp[5]) == 1 else False))
            i += img_gt_size
            data_list.append(
                dict(
                    img_path=osp.join(self.data_prefix['img'], filename),
                    instances=instances,
                ))

        # add image metas to data list
        img_metas = load(
            self.meta_file, file_format='pkl', backend_args=self.backend_args)
        assert len(img_metas) == len(data_list)
        for i, meta in enumerate(img_metas):
            img_id = osp.split(data_list[i]['img_path'])[-1][:-4]
            assert img_id == osp.split(meta['filename'])[-1][:-4]
            h, w = meta['ori_shape'][:2]
            data_list[i]['height'] = h
            data_list[i]['width'] = w
            data_list[i]['img_id'] = img_id
            # denormalize bboxes
            for j in range(len(data_list[i]['instances'])):
                data_list[i]['instances'][j]['bbox'][0] *= w
                data_list[i]['instances'][j]['bbox'][2] *= w
                data_list[i]['instances'][j]['bbox'][1] *= h
                data_list[i]['instances'][j]['bbox'][3] *= h
            # add image-level annotation
            if img_level_anns is not None:
                img_labels = []
                confidences = []
                img_ann_list = img_level_anns.get(img_id, [])
                for ann in img_ann_list:
                    img_labels.append(int(ann['image_level_label']))
                    confidences.append(float(ann['confidence']))
                data_list[i]['image_level_labels'] = np.array(
                    img_labels, dtype=np.int64)
                data_list[i]['confidences'] = np.array(
                    confidences, dtype=np.float32)
        return data_list

    def _parse_label_file(self, label_file: str) -> tuple:
        """Get classes name and index mapping from cls-label-description file.

        Args:
            label_file (str): File path of the label description file that
                maps the classes names in MID format to their short
                descriptions.

        Returns:
            tuple: Class name of OpenImages.
        """
        label_list = []
        id_list = []
        index_mapping = {}
        with get_local_path(
                label_file, backend_args=self.backend_args) as local_path:
            with open(local_path, 'r') as f:
                reader = csv.reader(f)
                for line in reader:
                    label_name = line[0]
                    label_id = int(line[2])
                    label_list.append(line[1])
                    id_list.append(label_id)
                    index_mapping[label_name] = label_id - 1
        indexes = np.argsort(id_list)
        classes_names = []
        for index in indexes:
            classes_names.append(label_list[index])
        return classes_names, index_mapping

    def _parse_img_level_ann(self, image_level_ann_file):
        """Parse image level annotations from csv style ann_file.

        Args:
            image_level_ann_file (str): CSV style image level annotation
                file path.

        Returns:
            defaultdict[list[dict]]: Annotations where item of the defaultdict
            indicates an image, each of which has (n) dicts.
            Keys of dicts are:

                - `image_level_label` (int): of shape 1.
                - `confidence` (float): of shape 1.
        """

        item_lists = defaultdict(list)
        with get_local_path(
                image_level_ann_file,
                backend_args=self.backend_args) as local_path:
            with open(local_path, 'r') as f:
                reader = csv.reader(f)
                i = -1
                for line in reader:
                    i += 1
                    if i == 0:
                        continue
                    else:
                        img_id = line[0]
                        label_id = line[1]
                        assert label_id in self.label_id_mapping
                        image_level_label = int(
                            self.label_id_mapping[label_id])
                        confidence = float(line[2])
                        item_lists[img_id].append(
                            dict(
                                image_level_label=image_level_label,
                                confidence=confidence))
        return item_lists

    def _get_relation_matrix(self, hierarchy_file: str) -> np.ndarray:
        """Get the matrix of class hierarchy from the hierarchy file.

        Args:
            hierarchy_file (str): File path to the hierarchy for classes.

        Returns:
            np.ndarray: The matrix of the corresponding
            relationship between the parent class and the child class,
            of shape (class_num, class_num).
        """
        with get_local_path(
                hierarchy_file, backend_args=self.backend_args) as local_path:
            class_label_tree = np.load(local_path, allow_pickle=True)
        return class_label_tree[1:, 1:]
