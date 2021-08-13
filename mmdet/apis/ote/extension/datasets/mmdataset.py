# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import json
import os
from copy import deepcopy
from typing import List

import numpy as np

from ote_sdk.entities.label import ScoredLabel
from ote_sdk.entities.shapes.box import Box

from sc_sdk.entities.annotation import Annotation, AnnotationScene, AnnotationSceneKind, NullMediaIdentifier
from sc_sdk.entities.datasets import Dataset, DatasetItem, NullDataset, Subset
from sc_sdk.entities.dataset_storage import NullDatasetStorage
from sc_sdk.entities.image import Image

from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import Compose


def get_annotation_mmdet_format(dataset_item: DatasetItem, label_list: List[str]) -> dict:
    """
    Function to convert a OTE annotation to mmdetection format. This is used both in the OTEDataset class defined in
    this file as in the custom pipeline element 'LoadAnnotationFromOTEDataset'

    :param dataset_item: DatasetItem for which to get annotations
    :param label_list: List of label names in the project
    :return dict: annotation information dict in mmdet format
    """
    width, height = dataset_item.width, dataset_item.height

    # load annotations for item
    gt_bboxes = []
    gt_labels = []

    for ann in dataset_item.get_annotations():
        box = ann.shape
        if not isinstance(box, Box):
            continue

        gt_bboxes.append([box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height])

        if ann.get_labels():
            # Label is not empty, add it to the gt labels
            label = ann.get_labels()[0]
            class_name = label.name
            gt_labels.append(label_list.index(class_name))
            is_empty_label = False
        else:
            is_empty_label = True

    if not ((len(gt_bboxes) == 1) and is_empty_label):
        ann_info = dict(
            bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
            labels=np.array(gt_labels, dtype=int)
        )
    else:
        ann_info = dict(bboxes=np.array([0, 0, 0, 0], dtype=np.float32).reshape(-1, 4),
                        labels=np.array([-1], dtype=int))
    return ann_info


@DATASETS.register_module()
class OTEDataset(CustomDataset):
    """
    Wrapper that allows using a OTE dataset to train mmdetection models. This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTE Dataset object.

    The wrapper overwrites some methods of the CustomDataset class: prepare_train_img, prepare_test_img and prepipeline
    Naming of certain attributes might seem a bit peculiar but this is due to the conventions set in CustomDataset. For
    instance, CustomDatasets expects the dataset items to be stored in the attribute data_infos, which is why it is
    named like that and not dataset_items.

    """

    class _DataInfoProxy:
        """
        This class is intended to be a wrapper to use it in CustomDataset-derived class as `self.data_infos`.
        Instead of using list `data_infos` as in CustomDataset, our implementation of dataset OTEDataset
        uses this proxy class with overriden __len__ and __getitem__; this proxy class
        forwards data access operations to ote_dataset and converts the dataset items to the view
        convenient for mmdetection.
        """
        def __init__(self, ote_dataset, classes):
            self.ote_dataset = ote_dataset
            self.CLASSES = classes

        def __len__(self):
            return len(self.ote_dataset)

        def __getitem__(self, index):
            """
            Prepare a dict 'data_info' that is expected by the mmdet pipeline to handle images and annotations
            :return data_info: dictionary that contains the image and image metadata, as well as the labels of the objects
                in the image
            """

            dataset = self.ote_dataset
            item = dataset[index]

            height, width = item.height, item.width

            data_info = dict(dataset_item=item, width=width, height=height, dataset_id=dataset.id, index=index,
                            ann_info=dict(label_list=self.CLASSES))

            return data_info

    def __init__(self, ote_dataset: Dataset, pipeline, classes=None, test_mode: bool = False):
        self.ote_dataset = ote_dataset
        self.test_mode = test_mode
        self.CLASSES = self.get_classes(classes)

        # Instead of using list data_infos as in CustomDataset, this implementation of dataset
        # uses a proxy class with overriden __len__ and __getitem__; this proxy class
        # forwards data access operations to ote_dataset.
        # Note that list `data_infos` cannot be used here, since OTE dataset class does not have interface to
        # get only annotation of a data item, so we would load the whole data item (including image)
        # even if we need only checking aspect ratio of the image; due to it
        # this implementation of dataset does not uses such tricks as skipping images with wrong aspect ratios or
        # small image size, since otherwise reading the whole dataset during initialization will be required.
        self.data_infos = OTEDataset._DataInfoProxy(ote_dataset, self.CLASSES)

        self.proposals = None  # Attribute expected by mmdet but not used for OTE datasets

        if not test_mode:
            self._set_group_flag()

        self.pipeline = Compose(pipeline)

    def _set_group_flag(self):
        """Set flag for grouping images.

        Originally, in Custom dataset, images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        This implementation will set group 0 for every image.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _rand_another(self, idx):
        return np.random.choice(len(self))

    # In contrast with CustomDataset this implementation of dataset
    # does not filter images w.r.t. the min size
    def _filter_imgs(self, min_size=32):
        raise NotImplementedError

    def prepare_train_img(self, idx: int) -> dict:
        """Get training data and annotations after pipeline.

        :param idx: int, Index of data.
        :return dict: Training data and annotation after pipeline with new keys introduced by pipeline.
        """
        item = deepcopy(self.data_infos[idx])
        self.pre_pipeline(item)
        return self.pipeline(item)

    def prepare_test_img(self, idx: int) -> dict:
        """Get testing data after pipeline.

        :param idx: int, Index of data.
        :return dict: Testing data after pipeline with new keys introduced by pipeline.
        """
        # FIXME.
        # item = deepcopy(self.data_infos[idx])
        item = self.data_infos[idx]
        self.pre_pipeline(item)
        return self.pipeline(item)

    @staticmethod
    def pre_pipeline(results: dict):
        """Prepare results dict for pipeline. Add expected keys to the dict. """
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def get_ann_info(self, idx):
        """
        This method is used for evaluation of predictions. The CustomDataset class implements a method
        CustomDataset.evaluate, which uses the class method get_ann_info to retrieve annotations.

        :param idx: index of the dataset item for which to get the annotations
        :return ann_info: dict that contains the coordinates of the bboxes and their corresponding labels
        """
        dataset_item = self.ote_dataset[idx]
        label_list = self.CLASSES
        if label_list is None:
            # For RepeatDataset wrapper.
            label_list = self.dataset.CLASSES
        return get_annotation_mmdet_format(dataset_item, label_list)


def get_classes_from_annotation(path):
    with open(path) as read_file:
        content = json.load(read_file)
        categories = [v['name'] for v in sorted(content['categories'], key=lambda x: x['id'])]
    return categories


class MMDatasetAdapter(Dataset):
    def __init__(self,
                 train_ann_file=None,
                 train_data_root=None,
                 val_ann_file=None,
                 val_data_root=None,
                 test_ann_file=None,
                 test_data_root=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.ann_files = {}
        self.data_roots = {}
        self.ann_files[Subset.TRAINING] = train_ann_file
        self.data_roots[Subset.TRAINING] = train_data_root
        self.ann_files[Subset.VALIDATION] = val_ann_file
        self.data_roots[Subset.VALIDATION] = val_data_root
        self.ann_files[Subset.TESTING] = test_ann_file
        self.data_roots[Subset.TESTING] = test_data_root
        self.coco_dataset = None
        for k, v in self.ann_files.items():
            if v:
                self.ann_files[k] = os.path.abspath(v)
        for k, v in self.data_roots.items():
            if v:
                self.data_roots[k] = os.path.abspath(v)
        self.labels = None
        self.set_labels_obtained_from_annotation()
        self.project_labels = None

    def set_labels_obtained_from_annotation(self):
        self.labels = None
        for subset in (Subset.TRAINING, Subset.VALIDATION, Subset.TESTING):
            path = self.ann_files[subset]
            if path:
                labels = get_classes_from_annotation(path)
                if self.labels and self.labels != labels:
                    raise RuntimeError('Labels are different from annotation file to annotation file.')
                self.labels = labels
        assert self.labels is not None

    def set_project_labels(self, project_labels):
        self.project_labels = project_labels

    def label_name_to_project_label(self, label_name):
        return [label for label in self.project_labels if label.name == label_name][0]

    def init_as_subset(self, subset: Subset):
        test_mode = subset in {Subset.VALIDATION, Subset.TESTING}
        if self.ann_files[subset] is None:
            return False
        from mmdet.datasets.pipelines import LoadImageFromFile, LoadAnnotations
        pipeline = [LoadImageFromFile(), LoadAnnotations(with_bbox=True)]
        self.coco_dataset = CocoDataset(ann_file=self.ann_files[subset],
                                        pipeline=pipeline,
                                        data_root=self.data_roots[subset],
                                        classes=self.labels,
                                        test_mode=test_mode)
        self.coco_dataset.test_mode = False
        return True

    def __getitem__(self, indx) -> dict:
        def create_gt_scored_label(label_name):
            return ScoredLabel(label=self.label_name_to_project_label(label_name))

        def create_gt_box(x1, y1, x2, y2, label):
            return Annotation(Box(x1=x1, y1=y1, x2=x2, y2=y2),
                              labels=[create_gt_scored_label(label)])

        item = self.coco_dataset[indx]
        divisor = np.tile([item['ori_shape'][:2][::-1]], 2)
        bboxes = item['gt_bboxes'] / divisor
        labels = item['gt_labels']

        shapes = [create_gt_box(*coords, self.labels[label_id]) for coords, label_id in zip(bboxes, labels)]

        image = Image(name=None, numpy=item['img'], dataset_storage=NullDatasetStorage())
        annotation_scene = AnnotationScene(kind=AnnotationSceneKind.ANNOTATION,
                                           media_identifier=NullMediaIdentifier(),
                                           annotations=shapes)
        datset_item = DatasetItem(image, annotation_scene)
        return datset_item

    def __len__(self) -> int:
        assert self.coco_dataset is not None
        return len(self.coco_dataset)

    def get_labels(self) -> list:
        return self.labels

    def get_subset(self, subset: Subset) -> Dataset:
        dataset = deepcopy(self)
        if dataset.init_as_subset(subset):
            return dataset
        return NullDataset()
