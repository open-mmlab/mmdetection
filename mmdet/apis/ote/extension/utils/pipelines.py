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

from mmdet.datasets.builder import PIPELINES

from ..datasets import get_annotation_mmdet_format


@PIPELINES.register_module()
class LoadImageFromOTEDataset:
    """
    Pipeline element that loads an image from a OTE Dataset on the fly. Can do conversion to float 32 if needed.

    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['dataset_item']: dataset_item from which to load the image
        results['dataset_id']: id of the dataset to which the item belongs
        results['index']: index of the item in the dataset

    :param to_float32: optional bool, True to convert images to fp32. defaults to False
    """

    def __init__(self, to_float32: bool = False):
        self.to_float32 = to_float32

    def __call__(self, results):
        dataset_item = results['dataset_item']
        img = dataset_item.numpy
        shape = img.shape

        assert img.shape[0] == results['height'], f"{img.shape[0]} != {results['height']}"
        assert img.shape[1] == results['width'], f"{img.shape[1]} != {results['width']}"

        filename = f"Dataset {results['dataset_id']}: Index {results['index']}"
        results['filename'] = filename
        results['ori_filename'] = filename
        results['img'] = img
        results['img_shape'] = shape
        results['ori_shape'] = shape
        # Set initial values for default meta_keys
        results['pad_shape'] = shape
        num_channels = 1 if len(shape) < 3 else shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']

        if self.to_float32:
            results['img'] = results['img'].astype(np.float32)

        return results


@PIPELINES.register_module()
class LoadAnnotationFromOTEDataset:
    """
    Pipeline element that loads an annotation from a OTE Dataset on the fly.

    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['dataset_item']: dataset_item from which to load the annotation
        results['ann_info']['label_list']: list of all labels in the project

    """

    def __init__(self, with_bbox: bool = True, with_label: bool = True, with_mask: bool = False, with_seg: bool = False,
                 poly2mask: bool = True, with_text: bool = False):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.with_text = with_text

    @staticmethod
    def _load_bboxes(results, ann_info):
        results['bbox_fields'].append('gt_bboxes')
        results['gt_bboxes'] = copy.deepcopy(ann_info['bboxes'])
        return results

    @staticmethod
    def _load_labels(results, ann_info):
        results['gt_labels'] = copy.deepcopy(ann_info['labels'])
        return results

    def __call__(self, results):
        dataset_item = results['dataset_item']
        label_list = results['ann_info']['label_list']
        ann_info = get_annotation_mmdet_format(dataset_item, label_list)
        # TODO. First only load bboxes, will extend to masks for semantic segmentation
        if self.with_bbox:
            results = self._load_bboxes(results, ann_info)
            if results is None or len(results['gt_bboxes']) == 0:
                return None
        if self.with_label:
            results = self._load_labels(results, ann_info)
        return results
