import random
import re
from collections import defaultdict
from typing import List

import numpy as np
import torch


def clean_name(name):
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def cat2ind(categories: List):
    ind_to_class = {0: '__background__'}
    index = 1
    for x in categories:
        ind_to_class[index] = x
        index += 1
    return ind_to_class


def check_for_positive_overflow(results,
                                ind_to_class,
                                tokenizer,
                                max_seq_length=256):

    # Check if we have too many positive labels
    # generate a caption by appending the positive labels
    instances_num = len(results['gt_bboxes'])
    positive_label_set = set()
    for i in range(instances_num):
        label_i = results['gt_bboxes_labels'][i] + 1
        positive_label_set.add(label_i)

    positive_label_list = list(positive_label_set)

    # random shuffule so we can sample different annotations
    # at different epochs
    random.shuffle(positive_label_list)

    kept_lables = []
    length = 0

    for index, label in enumerate(positive_label_list):

        label_text = clean_name(ind_to_class[label]) + '. '  # "dog. "

        tokenized = tokenizer.tokenize(label_text)

        length += len(tokenized)

        if length > max_seq_length:
            # there could not be overflow for COCO dataset
            break
        else:
            kept_lables.append(label)

    keep_box_index = []
    for i in range(instances_num):
        label_i = results['gt_bboxes_labels'][
            i] + 1  # "+1" for mapping 0~79 to 1~80
        if label_i in kept_lables:
            keep_box_index.append(i)

    # keep_box_index = torch.LongTensor(keep_box_index)
    results['gt_bboxes'] = results['gt_bboxes'][keep_box_index]
    results['gt_masks'] = results['gt_masks'][keep_box_index]
    results['gt_bboxes_labels'] = results['gt_bboxes_labels'][keep_box_index]
    results['gt_instances_ids'] = results['gt_instances_ids'][keep_box_index]
    return results, length


def create_queries_and_maps(label_list, tokenizer, separation_tokens='. '):
    labels = list(range(1, len(label_list) + 1))  # [1, 2, ..., 80]

    # Clean label list
    label_list = [clean_name(i) for i in label_list]
    # Form the query and get the mapping
    tokens_positive = []
    start_i = 0
    end_i = 0
    objects_query = ''

    # sep between tokens, follow training
    separation_tokens = '. '

    for _index, label in enumerate(label_list):

        start_i = len(objects_query)

        objects_query += label

        end_i = len(objects_query)
        tokens_positive.append([(start_i, end_i)
                                ])  # Every label has a [(start, end)]

        if _index != len(label_list) - 1:
            objects_query += separation_tokens

    tokenized = tokenizer(objects_query, return_tensors='pt')

    # Create the mapping between tokenized sentence and the original label
    positive_map_token_to_label, positive_map_label_to_token = \
        create_positive_dict(
            tokenized, tokens_positive,
            labels=labels)  # from token position to original label
    return objects_query, positive_map_label_to_token


def create_positive_dict(tokenized, tokens_positive, labels):
    """construct a dictionary such that positive_map[i] = j,
        iff token i is mapped to j label"""
    positive_map = defaultdict(int)

    # Additionally, have positive_map_label_to_tokens
    positive_map_label_to_token = {}

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:  # noqa
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:  # noqa
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map_label_to_token[labels[j]] = []
            for i in range(beg_pos, end_pos + 1):
                positive_map[i] = labels[j]
                positive_map_label_to_token[labels[j]].append(i)
    return positive_map, positive_map_label_to_token


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is
    associated to token j."""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):  # loop over each object
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:  # noqa
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:  # noqa
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos:end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


class ConvertCocoPolysToMask(object):

    def __init__(self, return_tokens=False, tokenizer=None, max_query_len=256):
        self.return_tokens = return_tokens  # True
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len

    def __call__(self, target):

        anno = target['annotations']
        caption = target['caption'] if 'caption' in target else None
        tokens_positive = [obj['tokens_positive'] for obj in anno]

        target = {}
        if caption is not None:
            target['caption'] = caption

        if tokens_positive is not None:
            target['tokens_positive'] = tokens_positive

        if self.return_tokens and self.tokenizer is not None:  # True
            tokenized = self.tokenizer(
                caption,
                return_tensors='pt',
                max_length=self.max_query_len,
                truncation=True)
            target['positive_map'] = create_positive_map(
                tokenized, target['tokens_positive'])

        return target


def generate_control_options_given_probabilities(control_probabilities,
                                                 full_positive, full_negative):

    outer_prob = random.random()

    probability_one_negative = control_probabilities[0]
    probability_one_positive = control_probabilities[1]
    probability_full = control_probabilities[2]  # 0.5
    probability_drop_positive = control_probabilities[3]

    assert (probability_drop_positive == 0)

    if outer_prob < probability_one_negative:
        # a. probability_one_negative: only give one negative class
        # to mimic evaluation (10%)
        num_negatives = 1
        num_positives = 0
    elif outer_prob < probability_one_positive + probability_one_negative:
        # b. probability_one_positive: only give one positive class to
        #  mimic evaluation (10%)
        num_negatives = 0
        num_positives = 1
    elif outer_prob < probability_full + probability_one_positive \
            + probability_one_negative:
        # prob 0.5
        # c. probability_full: add both all positive and all negatives (20%)
        num_negatives = full_negative
        num_positives = full_positive
    else:  # prob 0.5
        if random.random() < 1.0:
            num_negatives = np.random.choice(max(1, full_negative)) + 1
        else:
            num_negatives = full_negative  # Full

        if random.random() < probability_drop_positive:  # False
            num_positives = np.random.choice(max(1, full_positive)) + 1
        else:
            num_positives = full_positive  # Full

    return num_negatives, num_positives


def convert_object_detection_to_grounding_optimized_for_od(
        results,
        ind_to_class,
        disable_shuffle=False,
        add_detection_prompt=False,
        add_detection_prompt_advanced=False,
        random_sample_negative=85,
        control_probabilities=(0.0, 0.0, 0.5, 0.0),
        restricted_negative_list=None,
        separation_tokens='. ',
        max_num_labels=-1,
        max_seq_length=256,
        tokenizer=None,
        positive_caption_length=0):
    '''
    ind_to_class: {0: "__background__", 1 : "person" ...}

    restricted_negative_list : for datasets with restricted negatives,
    sample only the negatives

    Convert object detection data into grounding data format, on the fly.

    Control options:
        1. add_detection_prompt: add "object detection : "
            to the front of the prompt
        2. num_negatives: randomly sampled negative classes
        3. num_positives: how many positives to keep (-1 means do not cut any)

    Probabilities to generate the control options:

        a. probability_one_negative: only give one negative class
            to mimic evaluation
        b. probability_one_positive: only give one positive class
            to mimic evaluation
        c. probability_full: add both all positive and all negatives
        d. other:
            randomly sample some negatives and some positives
            The below control options are independent of each other:
            - probability_random_negative: probability of randomly
              sample X negatives
            - probability_random_positive: probability of randomly
              sample some positives
    '''
    if restricted_negative_list is None:  # True
        valid_negative_indexes = list(ind_to_class.keys())  # [0, 1, 2, ... 80]
    else:
        valid_negative_indexes = restricted_negative_list

    def generate_senetence_given_labels(positive_label_list,
                                        negative_label_list,
                                        prompt_engineer_version='v2',
                                        disable_shuffle=False):

        label_to_positions = {}

        assert (prompt_engineer_version == 'v2')
        num_negatives = len(negative_label_list)
        num_positives = len(positive_label_list)
        label_list = negative_label_list + positive_label_list
        if not disable_shuffle:  # True
            random.shuffle(label_list)

        if add_detection_prompt:  # False
            if add_detection_prompt_advanced and (
                    num_negatives == 0
                    or num_positives == 0) and not disable_shuffle:
                pheso_caption = 'object detection query : '
            else:
                pheso_caption = 'object detection : '
        else:
            pheso_caption = ''

        for index, label in enumerate(label_list):

            start_index = len(pheso_caption)

            pheso_caption += clean_name(
                ind_to_class[label])  # NOTE: slight change...
            end_index = len(pheso_caption)

            label_to_positions[label] = [start_index, end_index]

            if index != len(label_list) - 1:
                pheso_caption += separation_tokens  # += ". "

        return label_to_positions, pheso_caption

    if disable_shuffle:  # False
        label_list = list(sorted(
            ind_to_class.keys()))[1:]  # do not include the background
        label_to_positions, pheso_caption = generate_senetence_given_labels(
            positive_label_list=label_list,
            negative_label_list=[],
            disable_shuffle=True)
        # print(label_to_positions, pheso_caption)
    else:
        positive_label_set = set()
        for i in range(len(results['gt_bboxes'])):
            label_i = results['gt_bboxes_labels'][i] + 1
            positive_label_set.add(label_i)

        full_positive = len(
            positive_label_set)  # num classes containing in the current image
        if max_num_labels <= 0:  # -1
            full_negative = random_sample_negative  # 85
        else:
            full_negative = max(
                min(max_num_labels - full_positive, random_sample_negative), 0)

        if full_negative > len(valid_negative_indexes):  # True (85 > 81)
            full_negative = len(valid_negative_indexes)  # 81

        num_negatives, num_positives = \
            generate_control_options_given_probabilities(
                control_probabilities=control_probabilities,
                full_positive=full_positive,
                full_negative=full_negative)
        # num_positives not used

        # Keep some negatives
        negative_label_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)
            for i in np.random.choice(
                    valid_negative_indexes, size=num_negatives, replace=False):
                # label_sets.add(i)
                if i not in positive_label_set:
                    negative_label_list.add(i)

        # Keep all positives; ignoring num_positives
        positive_label_list = list(positive_label_set)
        random.shuffle(positive_label_list)

        negative_label_list = list(
            negative_label_list
        )  # e.g.: [17, 1, 13] where each number is the class name
        random.shuffle(negative_label_list)

        negative_max_length = max_seq_length - positive_caption_length
        screened_negative_label_list = []
        for negative_label in negative_label_list:
            label_text = clean_name(
                ind_to_class[negative_label]) + '. '  # "dog. "

            tokenized = tokenizer.tokenize(label_text)

            negative_max_length -= len(tokenized)

            if negative_max_length > 0:
                screened_negative_label_list.append(
                    negative_label)  # keep this negative
            else:
                break
        negative_label_list = screened_negative_label_list

        label_to_positions, pheso_caption = generate_senetence_given_labels(
            positive_label_list=positive_label_list,
            negative_label_list=negative_label_list)
    new_target = []
    # label_to_positions: dict
    # key: class index (range from 0-80)
    # value: their (char-level) positions in the caption
    for i in range(len(results['gt_bboxes'])):
        new_target_i = {}
        label_i = results['gt_bboxes_labels'][i] + 1
        if label_i in label_to_positions:
            new_target_i['tokens_positive'] = [label_to_positions[label_i]]
            new_target.append(new_target_i)
    return new_target, pheso_caption, label_to_positions
