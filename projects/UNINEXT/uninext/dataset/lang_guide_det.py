from typing import List

import numpy as np
from mmcv.transforms import BaseTransform
from transformers import AutoTokenizer, RobertaTokenizerFast

from mmdet.registry import TRANSFORMS
from .lang_utils import (
    ConvertCocoPolysToMask, cat2ind, check_for_positive_overflow,
    convert_object_detection_to_grounding_optimized_for_od,
    create_queries_and_maps)


@TRANSFORMS.register_module()
class LangGuideDet(BaseTransform):

    def __init__(self,
                 class_name: List = None,
                 train_state: bool = False,
                 use_roberta: bool = False,
                 max_query_len: int = 256):
        self.ind_to_class = cat2ind(class_name)
        if use_roberta:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(
                'projects/UNINEXT/uninext/roberta-base')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                'projects/UNINEXT/uninext/bert-base-uncased')

        self.train_state = train_state
        self.max_query_len = max_query_len
        self.prepare = ConvertCocoPolysToMask(
            return_tokens=True,
            tokenizer=self.tokenizer,
            max_query_len=self.max_query_len)

        prompt_test, positive_map_label_to_token = create_queries_and_maps(
            class_name, self.tokenizer)
        self.prompt_test = prompt_test
        self.positive_map_label_to_token = positive_map_label_to_token
        self.ordinal_nums = [
            'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
            'eighth', 'ninth', 'tenth'
        ]

    def transform(self, results: dict):
        """Transform the video information.

        Args:
            results (dict): The whole video information.
        """
        if self.train_state:
            original_box_num = len(results['gt_bboxes'])
            results, positive_caption_length = check_for_positive_overflow(
                results, self.ind_to_class, self.tokenizer,
                self.max_query_len - 2)
            new_box_num = len(results['gt_bboxes'])
            if new_box_num < original_box_num:
                print('WARNING: removed {} boxes due to positive caption'
                      'overflow'.format(original_box_num - new_box_num))
            annotations, caption, label_to_positions = \
                convert_object_detection_to_grounding_optimized_for_od(
                    results=results,
                    ind_to_class=self.ind_to_class,
                    positive_caption_length=positive_caption_length,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.max_query_len - 2)
            anno = {
                'annotations': annotations,
                'caption': caption,
                'label_to_positions': label_to_positions
            }
            anno = self.prepare(anno)
            results['positive_map'] = anno['positive_map'].bool()
            results['expressions'] = anno['caption']
        else:
            results['expressions'] = self.prompt_test
            results[
                'positive_map_label_to_token'] = \
                self.positive_map_label_to_token
            results['positive_map'] = np.array([], dtype=np.float32)

        return results
