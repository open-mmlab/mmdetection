# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

from pycocotools.coco import COCO

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset


def convert_phrase_ids(phrase_ids: list) -> list:
    unique_elements = sorted(set(phrase_ids))
    element_to_new_label = {
        element: label
        for label, element in enumerate(unique_elements)
    }
    phrase_ids = [element_to_new_label[element] for element in phrase_ids]
    return phrase_ids


@DATASETS.register_module()
class Flickr30kDataset(BaseDetDataset):
    """Flickr30K Dataset."""

    def load_data_list(self) -> List[dict]:

        self.coco = COCO(self.ann_file)

        self.ids = sorted(list(self.coco.imgs.keys()))

        data_list = []
        for img_id in self.ids:
            if isinstance(img_id, str):
                ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            else:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)

            coco_img = self.coco.loadImgs(img_id)[0]

            caption = coco_img['caption']
            file_name = coco_img['file_name']
            img_path = osp.join(self.data_prefix['img'], file_name)
            width = coco_img['width']
            height = coco_img['height']
            tokens_positive = coco_img['tokens_positive_eval']
            phrases = [caption[i[0][0]:i[0][1]] for i in tokens_positive]
            phrase_ids = []

            instances = []
            annos = self.coco.loadAnns(ann_ids)
            for anno in annos:
                instance = {
                    'bbox': [
                        anno['bbox'][0], anno['bbox'][1],
                        anno['bbox'][0] + anno['bbox'][2],
                        anno['bbox'][1] + anno['bbox'][3]
                    ],
                    'bbox_label':
                    anno['category_id'],
                    'ignore_flag':
                    anno['iscrowd']
                }
                phrase_ids.append(anno['phrase_ids'])
                instances.append(instance)

            phrase_ids = convert_phrase_ids(phrase_ids)

            data_list.append(
                dict(
                    img_path=img_path,
                    img_id=img_id,
                    height=height,
                    width=width,
                    instances=instances,
                    text=caption,
                    phrase_ids=phrase_ids,
                    tokens_positive=tokens_positive,
                    phrases=phrases,
                ))

        return data_list
