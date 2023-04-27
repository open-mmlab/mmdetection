# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset

try:
    from dsdl.dataset import DSDLDataset
except ImportError:
    DSDLDataset = None


@DATASETS.register_module()
class DSDLDetDataset(BaseDetDataset):
    """Dataset for dsdl detection.

    Args:
        with_bbox(bool): Load bbox or not, defaults to be True.
        with_polygon(bool): Load polygon or not, defaults to be False.
        with_mask(bool): Load seg map mask or not, defaults to be False.
        with_imagelevel_label(bool): Load image level label or not,
            defaults to be False.
        with_hierarchy(bool): Load hierarchy information or not,
            defaults to be False.
        specific_key_path(dict): Path of specific key which can not
            be loaded by it's field name.
        pre_transform(dict): pre-transform functions before loading.
    """

    METAINFO = {}

    def __init__(self,
                 with_bbox: bool = True,
                 with_polygon: bool = False,
                 with_mask: bool = False,
                 with_imagelevel_label: bool = False,
                 with_hierarchy: bool = False,
                 specific_key_path: dict = {},
                 pre_transform: dict = {},
                 **kwargs) -> None:

        if DSDLDataset is None:
            raise RuntimeError(
                'Package dsdl is not installed. Please run "pip install dsdl".'
            )

        self.with_hierarchy = with_hierarchy
        self.specific_key_path = specific_key_path

        loc_config = dict(type='LocalFileReader', working_dir='')
        if kwargs.get('data_root'):
            kwargs['ann_file'] = os.path.join(kwargs['data_root'],
                                              kwargs['ann_file'])
        self.required_fields = ['Image', 'ImageShape', 'Label', 'ignore_flag']
        if with_bbox:
            self.required_fields.append('Bbox')
        if with_polygon:
            self.required_fields.append('Polygon')
        if with_mask:
            self.required_fields.append('LabelMap')
        if with_imagelevel_label:
            self.required_fields.append('image_level_labels')
            assert 'image_level_labels' in specific_key_path.keys(
            ), '`image_level_labels` not specified in `specific_key_path` !'

        self.extra_keys = [
            key for key in self.specific_key_path.keys()
            if key not in self.required_fields
        ]

        self.dsdldataset = DSDLDataset(
            dsdl_yaml=kwargs['ann_file'],
            location_config=loc_config,
            required_fields=self.required_fields,
            specific_key_path=specific_key_path,
            transform=pre_transform,
        )

        BaseDetDataset.__init__(self, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load data info from an dsdl yaml file named as ``self.ann_file``

        Returns:
            List[dict]: A list of data info.
        """
        if self.with_hierarchy:
            # get classes_names and relation_matrix
            classes_names, relation_matrix = \
                self.dsdldataset.class_dom.get_hierarchy_info()
            self._metainfo['classes'] = tuple(classes_names)
            self._metainfo['RELATION_MATRIX'] = relation_matrix

        else:
            self._metainfo['classes'] = tuple(self.dsdldataset.class_names)

        data_list = []

        for i, data in enumerate(self.dsdldataset):
            # basic image info, including image id, path and size.
            datainfo = dict(
                img_id=i,
                img_path=os.path.join(self.data_prefix['img_path'],
                                      data['Image'][0].location),
                width=data['ImageShape'][0].width,
                height=data['ImageShape'][0].height,
            )

            # get image label info
            if 'image_level_labels' in data.keys():
                if self.with_hierarchy:
                    # get leaf node name when using hierarchy classes
                    datainfo['image_level_labels'] = [
                        self._metainfo['classes'].index(i.leaf_node_name)
                        for i in data['image_level_labels']
                    ]
                else:
                    datainfo['image_level_labels'] = [
                        self._metainfo['classes'].index(i.name)
                        for i in data['image_level_labels']
                    ]

            # get semantic segmentation info
            if 'LabelMap' in data.keys():
                datainfo['seg_map_path'] = data['LabelMap']

            # load instance info
            instances = []
            if 'Bbox' in data.keys():
                for idx in range(len(data['Bbox'])):
                    bbox = data['Bbox'][idx]
                    if self.with_hierarchy:
                        # get leaf node name when using hierarchy classes
                        label = data['Label'][idx].leaf_node_name
                        label_index = self._metainfo['classes'].index(label)
                    else:
                        label = data['Label'][idx].name
                        label_index = self._metainfo['classes'].index(label)

                    instance = {}
                    instance['bbox'] = bbox.xyxy
                    instance['bbox_label'] = label_index

                    if 'ignore_flag' in data.keys():
                        # get ignore flag
                        instance['ignore_flag'] = data['ignore_flag'][idx]
                    else:
                        instance['ignore_flag'] = 0

                    if 'Polygon' in data.keys():
                        # get polygon info
                        polygon = data['Polygon'][idx]
                        instance['mask'] = polygon.openmmlabformat

                    for key in self.extra_keys:
                        # load extra instance info
                        instance[key] = data[key][idx]

                    instances.append(instance)

            datainfo['instances'] = instances
            # append a standard sample in data list
            if len(datainfo['instances']) > 0:
                data_list.append(datainfo)

        return data_list

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

        valid_data_list = []
        for i, data_info in enumerate(self.data_list):
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            if min(width, height) >= min_size:
                valid_data_list.append(data_info)

        return valid_data_list
