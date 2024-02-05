# Copyright (c) OpenMMLab. All rights reserved.
import json

from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes

try:
    from transformers import AutoTokenizer
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

import random
import re

import numpy as np


def clean_name(name):
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    name = name.lower()
    return name


def check_for_positive_overflow(gt_bboxes, gt_labels, text, tokenizer,
                                max_tokens):
    # Check if we have too many positive labels
    # generate a caption by appending the positive labels
    positive_label_list = np.unique(gt_labels).tolist()
    # random shuffule so we can sample different annotations
    # at different epochs
    random.shuffle(positive_label_list)

    kept_lables = []
    length = 0

    for index, label in enumerate(positive_label_list):

        label_text = clean_name(text[str(label)]) + '. '

        tokenized = tokenizer.tokenize(label_text)

        length += len(tokenized)

        if length > max_tokens:
            break
        else:
            kept_lables.append(label)

    keep_box_index = []
    keep_gt_labels = []
    for i in range(len(gt_labels)):
        if gt_labels[i] in kept_lables:
            keep_box_index.append(i)
            keep_gt_labels.append(gt_labels[i])

    return gt_bboxes[keep_box_index], np.array(
        keep_gt_labels, dtype=np.long), length


def generate_senetence_given_labels(positive_label_list, negative_label_list,
                                    text):
    label_to_positions = {}

    label_list = negative_label_list + positive_label_list

    random.shuffle(label_list)

    pheso_caption = ''

    label_remap_dict = {}
    for index, label in enumerate(label_list):

        start_index = len(pheso_caption)

        pheso_caption += clean_name(text[str(label)])

        end_index = len(pheso_caption)

        if label in positive_label_list:
            label_to_positions[index] = [[start_index, end_index]]
            label_remap_dict[int(label)] = index

        # if index != len(label_list) - 1:
        #     pheso_caption += '. '
        pheso_caption += '. '

    return label_to_positions, pheso_caption, label_remap_dict


@TRANSFORMS.register_module()
class RandomSamplingNegPos(BaseTransform):

    def __init__(self,
                 tokenizer_name,
                 num_sample_negative=85,
                 max_tokens=256,
                 full_sampling_prob=0.5,
                 label_map_file=None):
        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_sample_negative = num_sample_negative
        self.full_sampling_prob = full_sampling_prob
        self.max_tokens = max_tokens
        self.label_map = None
        if label_map_file:
            with open(label_map_file, 'r') as file:
                self.label_map = json.load(file)

    def transform(self, results: dict) -> dict:
        if 'phrases' in results:
            return self.vg_aug(results)
        else:
            return self.od_aug(results)

    def vg_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        text = results['text'].lower().strip()
        if not text.endswith('.'):
            text = text + '. '

        phrases = results['phrases']
        # TODO: add neg
        positive_label_list = np.unique(gt_labels).tolist()
        label_to_positions = {}
        for label in positive_label_list:
            label_to_positions[label] = phrases[label]['tokens_positive']

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = text
        results['tokens_positive'] = label_to_positions
        return results

    def od_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']

        if 'text' not in results:
            assert self.label_map is not None
            text = self.label_map
        else:
            text = results['text']

        original_box_num = len(gt_labels)
        # If the category name is in the format of 'a/b' (in object365),
        # we randomly select one of them.
        for key, value in text.items():
            if '/' in value:
                text[key] = random.choice(value.split('/')).strip()

        gt_bboxes, gt_labels, positive_caption_length = \
            check_for_positive_overflow(gt_bboxes, gt_labels,
                                        text, self.tokenizer, self.max_tokens)

        if len(gt_bboxes) < original_box_num:
            print('WARNING: removed {} boxes due to positive caption overflow'.
                  format(original_box_num - len(gt_bboxes)))

        valid_negative_indexes = list(text.keys())

        positive_label_list = np.unique(gt_labels).tolist()
        full_negative = self.num_sample_negative

        if full_negative > len(valid_negative_indexes):
            full_negative = len(valid_negative_indexes)

        outer_prob = random.random()

        if outer_prob < self.full_sampling_prob:
            # c. probability_full: add both all positive and all negatives
            num_negatives = full_negative
        else:
            if random.random() < 1.0:
                num_negatives = np.random.choice(max(1, full_negative)) + 1
            else:
                num_negatives = full_negative

        # Keep some negatives
        negative_label_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)

            for i in np.random.choice(
                    valid_negative_indexes, size=num_negatives, replace=False):
                if int(i) not in positive_label_list:
                    negative_label_list.add(i)

        random.shuffle(positive_label_list)

        negative_label_list = list(negative_label_list)
        random.shuffle(negative_label_list)

        negative_max_length = self.max_tokens - positive_caption_length
        screened_negative_label_list = []

        for negative_label in negative_label_list:
            label_text = clean_name(text[str(negative_label)]) + '. '

            tokenized = self.tokenizer.tokenize(label_text)

            negative_max_length -= len(tokenized)

            if negative_max_length > 0:
                screened_negative_label_list.append(negative_label)
            else:
                break
        negative_label_list = screened_negative_label_list
        label_to_positions, pheso_caption, label_remap_dict = \
            generate_senetence_given_labels(positive_label_list,
                                            negative_label_list, text)

        # label remap
        if len(gt_labels) > 0:
            gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_labels)

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = pheso_caption
        results['tokens_positive'] = label_to_positions

        return results


@TRANSFORMS.register_module()
class LoadTextAnnotations(BaseTransform):

    def transform(self, results: dict) -> dict:
        if 'phrases' in results:
            tokens_positive = [
                phrase['tokens_positive']
                for phrase in results['phrases'].values()
            ]
            results['tokens_positive'] = tokens_positive
        else:
            text = results['text']
            results['text'] = list(text.values())
        return results
