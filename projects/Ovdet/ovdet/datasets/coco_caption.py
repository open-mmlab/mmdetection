import os.path as osp

from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class CocoCaptionOVDDataset(CocoDataset):
    """Renamed from `CocoCaptionDataset' to avoid conflicts with the mmdet."""

    def prepare_data(self, idx):
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        if data_info['has_caption']:
            return self.pipeline(data_info)
        else:
            return None

    def parse_data_info(self, raw_data_info: dict):
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        data_info['captions'] = [
            img_info.get('captions', []),
        ]
        pos_cat_ids = img_info.get('pos_category_ids', [])
        tags = [self.cat2label[cat_id] for cat_id in pos_cat_ids]
        data_info['tags'] = [
            tags,
        ]
        data_info['image_ids'] = [
            img_info['img_id'],
        ]

        has_caption = len(img_info.get('captions', [])) > 0
        data_info['has_caption'] = has_caption

        instance = {}
        bbox = [0.0, 0.0, img_info['width'], img_info['height']]
        instance['ignore_flag'] = 0
        instance['bbox'] = bbox
        instance['bbox_label'] = 0

        data_info['instances'] = [instance]
        return data_info
