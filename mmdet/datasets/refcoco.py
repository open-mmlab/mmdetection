# Copyright (c) OpenMMLab. All rights reserved.
import collections
import os.path as osp
import random
from typing import Dict, List

import mmengine
from mmengine.dataset import BaseDataset

from mmdet.registry import DATASETS


@DATASETS.register_module()
class RefCocoDataset(BaseDataset):
    """RefCOCO dataset.

    The `Refcoco` and `Refcoco+` dataset is based on
    `ReferItGame: Referring to Objects in Photographs of Natural Scenes
    <http://tamaraberg.com/papers/referit.pdf>`_.

    The `Refcocog` dataset is based on
    `Generation and Comprehension of Unambiguous Object Descriptions
    <https://arxiv.org/abs/1511.02283>`_.

    Args:
        ann_file (str): Annotation file path.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str): Prefix for training data.
        split_file (str): Split file path.
        split (str): Split name. Defaults to 'train'.
        text_mode (str): Text mode. Defaults to 'random'.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 split_file: str,
                 data_prefix: Dict,
                 split: str = 'train',
                 text_mode: str = 'random',
                 **kwargs):
        self.split_file = split_file
        self.split = split

        assert text_mode in ['original', 'random', 'concat', 'select_first']
        self.text_mode = text_mode
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            **kwargs,
        )

    def _join_prefix(self):
        if not mmengine.is_abs(self.split_file) and self.split_file:
            self.split_file = osp.join(self.data_root, self.split_file)

        return super()._join_prefix()

    def _init_refs(self):
        """Initialize the refs for RefCOCO."""
        anns, imgs = {}, {}
        for ann in self.instances['annotations']:
            anns[ann['id']] = ann
        for img in self.instances['images']:
            imgs[img['id']] = img

        refs, ref_to_ann = {}, {}
        for ref in self.splits:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            # add mapping related to ref
            refs[ref_id] = ref
            ref_to_ann[ref_id] = anns[ann_id]

        self.refs = refs
        self.ref_to_ann = ref_to_ann

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        self.splits = mmengine.load(self.split_file, file_format='pkl')
        self.instances = mmengine.load(self.ann_file, file_format='json')
        self._init_refs()
        img_prefix = self.data_prefix['img_path']

        ref_ids = [
            ref['ref_id'] for ref in self.splits if ref['split'] == self.split
        ]
        full_anno = []
        for ref_id in ref_ids:
            ref = self.refs[ref_id]
            ann = self.ref_to_ann[ref_id]
            ann.update(ref)
            full_anno.append(ann)

        image_id_list = []
        final_anno = {}
        for anno in full_anno:
            image_id_list.append(anno['image_id'])
            final_anno[anno['ann_id']] = anno
        annotations = [value for key, value in final_anno.items()]

        coco_train_id = []
        image_annot = {}
        for i in range(len(self.instances['images'])):
            coco_train_id.append(self.instances['images'][i]['id'])
            image_annot[self.instances['images'][i]
                        ['id']] = self.instances['images'][i]

        images = []
        for image_id in list(set(image_id_list)):
            images += [image_annot[image_id]]

        data_list = []

        grounding_dict = collections.defaultdict(list)
        for anno in annotations:
            image_id = int(anno['image_id'])
            grounding_dict[image_id].append(anno)

        join_path = mmengine.fileio.get_file_backend(img_prefix).join_path
        for image in images:
            img_id = image['id']
            instances = []
            sentences = []
            for grounding_anno in grounding_dict[img_id]:
                texts = [x['raw'].lower() for x in grounding_anno['sentences']]
                # random select one text
                if self.text_mode == 'random':
                    idx = random.randint(0, len(texts) - 1)
                    text = [texts[idx]]
                # concat all texts
                elif self.text_mode == 'concat':
                    text = [''.join(texts)]
                # select the first text
                elif self.text_mode == 'select_first':
                    text = [texts[0]]
                # use all texts
                elif self.text_mode == 'original':
                    text = texts
                else:
                    raise ValueError(f'Invalid text mode "{self.text_mode}".')
                ins = [{
                    'mask': grounding_anno['segmentation'],
                    'ignore_flag': 0
                }] * len(text)
                instances.extend(ins)
                sentences.extend(text)
            data_info = {
                'img_path': join_path(img_prefix, image['file_name']),
                'img_id': img_id,
                'instances': instances,
                'text': sentences
            }
            data_list.append(data_info)

        if len(data_list) == 0:
            raise ValueError(f'No sample in split "{self.split}".')

        return data_list
