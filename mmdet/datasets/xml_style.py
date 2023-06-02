# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET
from typing import List, Optional, Union

import mmcv
from mmengine.fileio import get, get_local_path, list_from_file

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class XMLDataset(BaseDetDataset):
    """XML dataset for detection.

    Args:
        img_subdir (str): Subdir where images are stored. Default: JPEGImages.
        ann_subdir (str): Subdir where annotations are. Default: Annotations.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 img_subdir: str = 'JPEGImages',
                 ann_subdir: str = 'Annotations',
                 **kwargs) -> None:
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        super().__init__(**kwargs)

    @property
    def sub_data_root(self) -> str:
        """Return the sub data root."""
        return self.data_prefix.get('sub_data_root', '')

    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        assert self._metainfo.get('classes', None) is not None, \
            '`classes` in `XMLDataset` can not be None.'
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self._metainfo['classes'])
        }

        data_list = []
        img_ids = list_from_file(self.ann_file, backend_args=self.backend_args)
        for img_id in img_ids:
            file_name = osp.join(self.img_subdir, f'{img_id}.jpg')
            xml_path = osp.join(self.sub_data_root, self.ann_subdir,
                                f'{img_id}.xml')

            raw_img_info = {}
            raw_img_info['img_id'] = img_id
            raw_img_info['file_name'] = file_name
            raw_img_info['xml_path'] = xml_path

            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)
        return data_list

    @property
    def bbox_min_size(self) -> Optional[str]:
        """Return the minimum size of bounding boxes in the images."""
        if self.filter_cfg is not None:
            return self.filter_cfg.get('bbox_min_size', None)
        else:
            return None

    def parse_data_info(self, img_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            img_info (dict): Raw image information, usually it includes
                `img_id`, `file_name`, and `xml_path`.

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        data_info = {}
        img_path = osp.join(self.sub_data_root, img_info['file_name'])
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['xml_path'] = img_info['xml_path']

        # deal with xml file
        with get_local_path(
                img_info['xml_path'],
                backend_args=self.backend_args) as local_path:
            raw_ann_info = ET.parse(local_path)
        root = raw_ann_info.getroot()
        size = root.find('size')
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        else:
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, backend='cv2')
            height, width = img.shape[:2]
            del img, img_bytes

        data_info['height'] = height
        data_info['width'] = width

        data_info['instances'] = self._parse_instance_info(
            raw_ann_info, minus_one=True)

        return data_info

    def _parse_instance_info(self,
                             raw_ann_info: ET,
                             minus_one: bool = True) -> List[dict]:
        """parse instance information.

        Args:
            raw_ann_info (ElementTree): ElementTree object.
            minus_one (bool): Whether to subtract 1 from the coordinates.
                Defaults to True.

        Returns:
            List[dict]: List of instances.
        """
        instances = []
        for obj in raw_ann_info.findall('object'):
            instance = {}
            name = obj.find('name').text
            if name not in self._metainfo['classes']:
                continue
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]

            # VOC needs to subtract 1 from the coordinates
            if minus_one:
                bbox = [x - 1 for x in bbox]

            ignore = False
            if self.bbox_min_size is not None:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.bbox_min_size or h < self.bbox_min_size:
                    ignore = True
            if difficult or ignore:
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[name]
            instances.append(instance)
        return instances

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False
        min_size = self.filter_cfg.get('min_size', 0) \
            if self.filter_cfg is not None else 0

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
