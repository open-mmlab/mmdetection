# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List

import mmengine
from mmengine.dataset import BaseDataset

from mmdet.registry import DATASETS


@DATASETS.register_module()
class Flickr30KDataset(BaseDataset):
    """Flickr30K Dataset
    """

    def load_data_list(self) -> List[dict]:
        """
        """
        anns = mmengine.load(self.ann_file, file_format='json')
        
        data_list = []
        
        for img in anns['images']:
            img_id = img['id']
            img_file_name = img['file_name']
            img_path = osp.join(self.data_prefix['img'], img_file_name)
            width = img['width']
            height = img['height']

            # get sentence
            text = img['caption']
            text_id = img['sentence_id']

            # get annotations
            instances = []
            for ann in anns['annotations']:
                if img_id == ann['image_id']:
                    instance = {}
                    instance['bbox'] = ann['bbox']
                    instance['bbox_label'] = ann['category_id']
                    instance['ignore_flag'] = ann['iscrowd']
                    instance['phrase_id'] = ann['phrase_ids']
                    instance['sentence_id'] = text_id
                    instances.append(instance)
            
            data_info = {
                'img_path': img_path,
                'img_id': img_id,
                'height': height,
                'width': width,
                'instances': instances,
                'text': text
            }

            data_list.append(data_info)

        return data_list
