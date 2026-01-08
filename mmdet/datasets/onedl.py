# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
from loguru import logger
from onedl.core import (InstanceSegmentationInstances, LabelMap,
                        LineDetectionInstances, ObjectDetectionInstances)
from onedl.core.utils import get_image_size
from onedl.datasets import ObjectDetectionDataset
from onedl.datasets.specializations.line_detection import LineDetectionDataset

from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS
from ._palette import DEFAULT_PALETTE


@DATASETS.register_module()
class OneDLDataset(BaseDetDataset):

    def __init__(  # type: ignore  # noqa: PLR0913
        self,
        *args,
        dataset_name: str,
        seg_map_suffix: str = '.png',
        proposal_file: Optional[str] = None,
        file_client_args: Optional[dict] = None,
        backend_args: Optional[dict] = None,
        return_classes: bool = False,
        min_bbox_area: int = 1,
        clip_bboxes: bool = True,
        shuffle: bool = True,
        **kwargs,
    ) -> None:
        self.dataset_name = dataset_name
        self.min_bbox_area = min_bbox_area
        self.clip_bboxes = clip_bboxes
        self.shuffle = shuffle

        super().__init__(
            *args,
            seg_map_suffix=seg_map_suffix,
            proposal_file=proposal_file,
            file_client_args=file_client_args,
            backend_args=backend_args,
            return_classes=return_classes,
            **kwargs,
        )

    def _to_mm_instance(self, instances: Union[ObjectDetectionInstances,
                                               InstanceSegmentationInstances],
                        label_map: LabelMap):
        mm_instances = []
        has_mask = isinstance(instances, InstanceSegmentationInstances)
        for annotation in instances:
            mm_instance = dict()

            mm_instance['bbox'] = np.array(annotation.bbox.as_xyxy())
            class_id = label_map.label_to_class_id(annotation.label)
            mm_instance['bbox_label'] = class_id
            mm_instance['ignore_flag'] = False
            if has_mask:
                # noinspection PyUnresolvedReferences
                mm_instance['mask'] = [
                    np.array(x)
                    for x in annotation.mask.as_polygon_mask().polygons
                ]
            mm_instances.append(mm_instance)
        return mm_instances

    def _preprocess_dataset(self, dataset):
        if self.clip_bboxes:
            logger.info(
                f"Clipping dataset '{self.dataset_name}' bboxes to image size."
            )
            dataset.clip_bboxes_to_image_size(inplace=True)

        if self.min_bbox_area > 0:
            logger.info(
                f"Filtering dataset '{self.dataset_name}' bboxes by surface "
                f'({self.min_bbox_area}<).')
            dataset.filter_by_sqrt_area(self.min_bbox_area, inplace=True)

    def load_data_list(self) -> List[dict]:
        from onedl.client import Client

        client = Client()
        dataset = client.datasets.load(
            self.dataset_name, pull_blobs=True)  # type: ignore
        if self.shuffle:
            # shuffle the dataset
            idx = list(range(len(dataset)))
            np.random.shuffle(idx)
            dataset = dataset[idx]

        dataset: ObjectDetectionDataset = dataset.object_detection
        label_map = dataset.label_map
        metainfo = {
            'classes': label_map.get_labels(),
            'palette': DEFAULT_PALETTE[:label_map.n_classes]
        }
        self._preprocess_dataset(dataset)
        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo[k] = v

        self.cat_ids = label_map.get_class_ids()

        data_infos = []
        cat_to_imgs = defaultdict(list)
        for i, (img_path, instances) in enumerate(
                zip(dataset.inputs.path_iterator(), dataset.targets)):
            width, height = get_image_size(img_path)
            mm_instances = self._to_mm_instance(instances, label_map)

            for instance in mm_instances:
                cat_to_imgs[instance['bbox_label']].append(i)

            data_infos.append(
                dict(
                    img_path=img_path,
                    img_id=i,
                    width=width,
                    height=height,
                    instances=mm_instances))
        self.cat_to_imgs = cat_to_imgs

        return data_infos

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list  # type: ignore

        if self.filter_cfg is None:
            return self.data_list  # type: ignore

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for _i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_to_imgs[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for _i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos


@DATASETS.register_module()
class OneDLLineInstancesDatasetAdapter(OneDLDataset):

    def _to_mm_instance(self, instances: Union[LineDetectionInstances],
                        label_map: LabelMap) -> list[dict]:
        mm_instances = []
        for annotation in instances:
            mm_instance = dict()

            mm_instance['line'] = annotation.line.points
            class_id = label_map.label_to_class_id(annotation.label)
            mm_instance['line_label'] = class_id
            mm_instance['visible'] = annotation.visible
            mm_instances.append(mm_instance)
        return mm_instances

    def load_data_list(self) -> List[dict]:
        from onedl.client import Client

        client = Client()
        dataset = client.datasets.load(
            self.dataset_name, pull_blobs=True)  # type: ignore
        dataset: LineDetectionDataset = dataset.line_detection
        label_map = dataset.label_map
        metainfo = {
            'classes': label_map.get_labels(),
            'palette': DEFAULT_PALETTE[:label_map.n_classes]
        }

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo[k] = v

        self.cat_ids = label_map.get_class_ids()

        if 'segmentation' not in dataset.columns:
            msg = (
                'Segmentation maps are required for line detection.'
                ' Use `dataset.generate_segmentation_maps()` to generate them.'
            )
            raise ValueError(msg)

        data_infos = []
        cat_to_imgs = defaultdict(list)
        for i, (img_path, instances, segmentation_path) in enumerate(
                zip(
                    dataset.inputs.path_iterator(),
                    dataset.targets,
                    dataset['segmentation'].path_iterator() if 'segmentation'
                    in dataset.columns else [None] * len(dataset),
                )):
            width, height = get_image_size(img_path)
            mm_instances = self._to_mm_instance(instances, label_map)

            for instance in mm_instances:
                cat_to_imgs[instance['line_label']].append(i)

            data_infos.append(
                dict(
                    img_path=img_path,
                    img_id=i,
                    width=width,
                    height=height,
                    instances=mm_instances,
                    segmentation_path=segmentation_path,
                ))
        self.cat_to_imgs = cat_to_imgs

        return data_infos
