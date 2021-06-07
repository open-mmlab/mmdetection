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

import copy

import numpy as np
from typing import List

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import Compose

from sc_sdk.entities.datasets import Dataset, DatasetItem
from sc_sdk.entities.shapes.box import Box


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

    for box in dataset_item.annotation.shapes:
        if not isinstance(box, Box):
            continue

        gt_bboxes.append([box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height])

        if box.get_labels():
            # Label is not empty, add it to the gt labels
            label = box.get_labels()[0]
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
    def __init__(self, ote_dataset: Dataset, pipeline, classes=None, test_mode: bool = False):
        self.ote_dataset = ote_dataset
        self.test_mode = test_mode
        self.CLASSES = self.get_classes(classes)
        self.data_infos = self.load_dataset_items()
        self.proposals = None  # Attribute expected by mmdet but not used for OTE datasets

        if not test_mode:
            self._set_group_flag()

        self.pipeline = Compose(pipeline)

    def prepare_train_img(self, idx: int) -> dict:
        """Get training data and annotations after pipeline.

        :param idx: int, Index of data.
        :return dict: Training data and annotation after pipeline with new keys introduced by pipeline.
        """
        item = copy.deepcopy(self.data_infos[idx])
        self.pre_pipeline(item)
        return self.pipeline(item)

    def prepare_test_img(self, idx: int) -> dict:
        """Get testing data after pipeline.

        :param idx: int, Index of data.
        :return dict: Testing data after pipeline with new keys introduced by pipeline.
        """
        # FIXME.
        # item = copy.deepcopy(self.data_infos[idx])
        item = self.data_infos[idx]
        self.pre_pipeline(item)
        return self.pipeline(item)

    @staticmethod
    def pre_pipeline(results: dict):
        """Prepare results dict for pipeline. Add expected keys to the dict. """
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def load_dataset_items(self) -> List:
        """
        Prepare a dict 'data_infos' that is expected by the mmdet pipeline to handle images and annotations

        :return data_infos: dictionary that contains the image and image metadata, as well as the labels of the objects
            in the image
        """
        dataset = self.ote_dataset
        dataset_items = []

        for ii, item in enumerate(dataset):
            height, width = item.height, item.width

            img_info = dict(dataset_item=item, width=width, height=height, dataset_id=dataset.id, index=ii,
                            ann_info=dict(label_list=self.CLASSES))
            dataset_items.append(img_info)

        return dataset_items

    def get_ann_info(self, idx):
        """
        This method is used for evaluation of predictions. The CustomDataset class implements a method
        CustomDataset.evaluate, which uses the class method get_ann_info to retrieve annotations.

        :param idx: index of the dataset item for which to get the annotations
        :return ann_info: dict that contains the coordinates of the bboxes and their corresponding labels
        """
        dataset_item = self.ote_dataset[idx]
        label_list = self.CLASSES
        return get_annotation_mmdet_format(dataset_item, label_list)
