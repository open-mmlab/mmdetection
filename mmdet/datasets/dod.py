# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Optional

import numpy as np

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset

try:
    from d_cube import D3
except ImportError:
    D3 = None
from .api_wrappers import COCO


@DATASETS.register_module()
class DODDataset(BaseDetDataset):

    def __init__(self,
                 *args,
                 data_root: Optional[str] = '',
                 data_prefix: dict = dict(img_path=''),
                 **kwargs) -> None:
        if D3 is None:
            raise ImportError(
                'Please install d3 by `pip install ddd-dataset`.')
        pkl_anno_path = osp.join(data_root, data_prefix['anno'])
        self.img_root = osp.join(data_root, data_prefix['img'])
        self.d3 = D3(self.img_root, pkl_anno_path)

        sent_infos = self.d3.load_sents()
        classes = tuple([sent_info['raw_sent'] for sent_info in sent_infos])
        super().__init__(
            *args,
            data_root=data_root,
            data_prefix=data_prefix,
            metainfo={'classes': classes},
            **kwargs)

    def load_data_list(self) -> List[dict]:
        coco = COCO(self.ann_file)
        data_list = []
        img_ids = self.d3.get_img_ids()
        for img_id in img_ids:
            data_info = {}

            img_info = self.d3.load_imgs(img_id)[0]
            file_name = img_info['file_name']
            img_path = osp.join(self.img_root, file_name)
            data_info['img_path'] = img_path
            data_info['img_id'] = img_id
            data_info['height'] = img_info['height']
            data_info['width'] = img_info['width']

            group_ids = self.d3.get_group_ids(img_ids=[img_id])
            sent_ids = self.d3.get_sent_ids(group_ids=group_ids)
            sent_list = self.d3.load_sents(sent_ids=sent_ids)
            text_list = [sent['raw_sent'] for sent in sent_list]
            ann_ids = coco.get_ann_ids(img_ids=[img_id])
            anno = coco.load_anns(ann_ids)

            data_info['text'] = text_list
            data_info['sent_ids'] = np.array([s for s in sent_ids])
            data_info['custom_entities'] = True

            instances = []
            for i, ann in enumerate(anno):
                instance = {}
                x1, y1, w, h = ann['bbox']
                bbox = [x1, y1, x1 + w, y1 + h]
                instance['ignore_flag'] = 0
                instance['bbox'] = bbox
                instance['bbox_label'] = ann['category_id'] - 1
                instances.append(instance)
            data_info['instances'] = instances
            data_list.append(data_info)
        return data_list
