# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from mmdet.structures.bbox import BaseBoxes

try:
    from transformers import AutoTokenizer, BertConfig
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

import random
import torch
import re
import numpy as np


def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name


def check_for_positive_overflow(gt_bboxes, gt_labels, text, tokenizer, max_tokens):
    # NOTE: Only call this function for OD data; DO NOT USE IT FOR GROUNDING DATA
    # NOTE: called only in coco_dt

    # Check if we have too many positive labels
    # generate a caption by appending the positive labels
    positive_label_list = np.unique(gt_labels).tolist()
    # random shuffule so we can sample different annotations at different epochs
    random.shuffle(positive_label_list)

    kept_lables = []
    length = 0

    for index, label in enumerate(positive_label_list):

        label_text = clean_name(text[str(label)]) + ". "

        tokenized = tokenizer.tokenize(label_text)

        length += len(tokenized)

        if length > max_tokens:
            break
        else:
            kept_lables.append(label)

    keep_box_index = []
    for i in range(len(gt_labels)):
        if gt_labels[i] in kept_lables:
            keep_box_index.append(i)

    return gt_bboxes[keep_box_index], np.asarray(kept_lables)


@TRANSFORMS.register_module()
class RandomSamplingNegPos(BaseTransform):
    def __init__(self, tokenizer_name, num_sample_negative=85, max_tokens=256):
        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_sample_negative = num_sample_negative
        self.max_tokens = max_tokens

    def transform(self, results: dict) -> dict:
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        original_box_num= len(gt_labels)
        text = results['text']
        gt_bboxes, gt_labels = check_for_positive_overflow(gt_bboxes, gt_labels, text, self.tokenizer, self.max_tokens)

        if len(gt_bboxes) < original_box_num:
            print("WARNING: removed {} boxes due to positive caption overflow".format(original_box_num - len(gt_bboxes)))

        valid_negative_indexes = list(text.keys())

        positive_label_list = np.unique(gt_labels).tolist()
        full_positive = len(positive_label_set)
        if max_num_labels <= 0:
            full_negative = random_sample_negative  # 85
        else:
            full_negative = max(min(max_num_labels - full_positive, random_sample_negative), 0)





