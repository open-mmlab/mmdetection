# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

from mmengine.fileio import get_local_path

from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS
from .api_wrappers import COCO


@DATASETS.register_module()
class MDETRStyleRefCocoDataset(BaseDetDataset):
    """RefCOCO dataset.

    Only support evaluation now.
    """

    def load_data_list(self) -> List[dict]:
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            coco = COCO(local_path)

        img_ids = coco.get_img_ids()

        data_infos = []
        for img_id in img_ids:
            raw_img_info = coco.load_imgs([img_id])[0]
            ann_ids = coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = coco.load_anns(ann_ids)

            data_info = {}
            img_path = osp.join(self.data_prefix['img'],
                                raw_img_info['file_name'])
            data_info['img_path'] = img_path
            data_info['img_id'] = img_id
            data_info['height'] = raw_img_info['height']
            data_info['width'] = raw_img_info['width']
            data_info['dataset_mode'] = raw_img_info['dataset_name']

            data_info['text'] = raw_img_info['caption']
            data_info['custom_entities'] = False
            data_info['tokens_positive'] = -1

            instances = []
            for i, ann in enumerate(raw_ann_info):
                instance = {}
                x1, y1, w, h = ann['bbox']
                bbox = [x1, y1, x1 + w, y1 + h]
                instance['bbox'] = bbox
                instance['bbox_label'] = ann['category_id']
                instance['ignore_flag'] = 0
                instances.append(instance)

            data_info['instances'] = instances
            data_infos.append(data_info)
        return data_infos
