import collections
import copy
import itertools
import logging
import os.path as osp
import string
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.core import text_eval
from .builder import DATASETS
from .coco import CocoDataset, ConcatenatedCocoDataset


@DATASETS.register_module()
class CocoWithTextDataset(CocoDataset):

    CLASSES = ('text')

    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        results['text_fields'] = []

    def __init__(self, alphabet='  ' + string.ascii_lowercase + string.digits, max_texts_num=0, *args, **kwargs):
        self.max_texts_num = max_texts_num
        super().__init__(*args, **kwargs)
        self.alphabet = alphabet
        self.max_text_len = 33
        self.EOS = 1

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = collections.Counter(_['image_id'] for _ in self.coco.anns.values())
        if self.max_texts_num > 0:
            ids_with_ann = {k for k, v in ids_with_ann.items() if v <= self.max_texts_num}
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_texts = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            text = ann['text']['transcription'] if ann['text']['legible'] else ''
            text = text.lower()
            assert not ann.get('iscrowd', False) == ann['text']['legible']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if self.min_size is not None:
                if w < self.min_size or h < self.min_size:
                    continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

                if ' ' in text:
                    text = []
                else:
                    text = [self.alphabet.find(l) for l in text]
                    if -1 in text:
                        text = []
                    else:
                        text.append(self.EOS)
                # while len(text) < self.max_text_len:
                #     text.append(-1)
                text = np.array(text)
                gt_texts.append(text)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_texts:
            gt_texts = np.array(gt_texts)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            texts=gt_texts)

        return ann


@DATASETS.register_module()
class ConcatenatedCocoWithTextDataset(CocoWithTextDataset, ConcatenatedCocoDataset):
    def __init__(self, concatenated_dataset):
        ConcatenatedCocoDataset.__init__(self, concatenated_dataset)
        self.max_texts_num = concatenated_dataset.datasets[0].max_texts_num
        self.alphabet = concatenated_dataset.datasets[0].alphabet
        self.max_text_len = concatenated_dataset.datasets[0].max_text_len
        self.EOS = concatenated_dataset.datasets[0].EOS

        assert all(self.max_texts_num == x.max_texts_num for x in concatenated_dataset.datasets)
        assert all(self.alphabet == x.alphabet for x in concatenated_dataset.datasets)
        assert all(self.max_text_len == x.max_text_len for x in concatenated_dataset.datasets)
        assert all(self.EOS == x.EOS for x in concatenated_dataset.datasets)
