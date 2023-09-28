# Copyright (c) OpenMMLab. All rights reserved.
import re
import warnings
from typing import Tuple, Union

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector


def find_noun_phrases(caption: str) -> list:
    """Find noun phrases in a caption using nltk.
    Args:
        caption (str): The caption to analyze.

    Returns:
        list: List of noun phrases found in the caption.

    Examples:
        >>> caption = 'There is two cat and a remote in the picture'
        >>> find_noun_phrases(caption) # ['cat', 'a remote', 'the picture']
    """
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
    except ImportError:
        raise RuntimeError('nltk is not installed, please install it by: '
                           'pip install nltk.')

    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = 'NP: {<DT>?<JJ.*>*<NN.*>+}'
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = []
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    return noun_phrases


def remove_punctuation(text: str) -> str:
    """Remove punctuation from a text.
    Args:
        text (str): The input text.

    Returns:
        str: The text with punctuation removed.
    """
    punctuation = [
        '|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^', '\'', '\"', '’',
        '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
    ]
    for p in punctuation:
        text = text.replace(p, '')
    return text.strip()


def run_ner(caption: str) -> Tuple[list, list]:
    """Run NER on a caption and return the tokens and noun phrases.
    Args:
        caption (str): The input caption.

    Returns:
        Tuple[List, List]: A tuple containing the tokens and noun phrases.
            - tokens_positive (List): A list of token positions.
            - noun_phrases (List): A list of noun phrases.
    """
    noun_phrases = find_noun_phrases(caption)
    noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
    noun_phrases = [phrase for phrase in noun_phrases if phrase != '']
    relevant_phrases = noun_phrases
    labels = noun_phrases

    tokens_positive = []
    for entity, label in zip(relevant_phrases, labels):
        try:
            # search all occurrences and mark them as different entities
            # TODO: Not Robust
            for m in re.finditer(entity, caption.lower()):
                tokens_positive.append([[m.start(), m.end()]])
        except Exception:
            print('noun entities:', noun_phrases)
            print('entity:', entity)
            print('caption:', caption.lower())
    return tokens_positive, noun_phrases


def create_positive_map(tokenized,
                        tokens_positive: list,
                        max_num_entities: int = 256) -> Tensor:
    """construct a map such that positive_map[i,j] = True
    if box i is associated to token j

    Args:
        tokenized: The tokenized input.
        tokens_positive (list): A list of token ranges
            associated with positive boxes.
        max_num_entities (int, optional): The maximum number of entities.
            Defaults to 256.

    Returns:
        torch.Tensor: The positive map.

    Raises:
        Exception: If an error occurs during token-to-char mapping.
    """
    positive_map = torch.zeros((len(tokens_positive), max_num_entities),
                               dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                print('beg:', beg, 'end:', end)
                print('token_positive:', tokens_positive)
                raise e
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except Exception:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except Exception:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos:end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def create_positive_map_label_to_token(positive_map: Tensor,
                                       plus: int = 0) -> dict:
    """Create a dictionary mapping the label to the token.
    Args:
        positive_map (Tensor): The positive map tensor.
        plus (int, optional): Value added to the label for indexing.
            Defaults to 0.

    Returns:
        dict: The dictionary mapping the label to the token.
    """
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(
            positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token


@MODELS.register_module()
class GLIP(SingleStageDetector):
    """Implementation of `GLIP <https://arxiv.org/abs/2112.03857>`_
    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        language_model (:obj:`ConfigDict` or dict): The language model config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of GLIP. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of GLIP. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 language_model: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.language_model = MODELS.build(language_model)

        self._special_tokens = '. '

    def get_tokens_and_prompts(
            self,
            original_caption: Union[str, list, tuple],
            custom_entities: bool = False) -> Tuple[dict, str, list, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            caption_string = ''
            tokens_positive = []
            for idx, word in enumerate(original_caption):
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
                if idx != len(original_caption) - 1:
                    caption_string += self._special_tokens
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            entities = original_caption
        else:
            original_caption = original_caption.strip(self._special_tokens)
            tokenized = self.language_model.tokenizer([original_caption],
                                                      return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(tokenized, tokens_positive)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
            self,
            original_caption: Union[str, list, tuple],
            custom_entities: bool = False) -> Tuple[dict, str, Tensor, list]:
        tokenized, caption_string, tokens_positive, entities = \
            self.get_tokens_and_prompts(
                original_caption, custom_entities)
        positive_map_label_to_token, positive_map = self.get_positive_map(
            tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        # TODO: Only open vocabulary tasks are supported for training now.
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        new_text_prompts = []
        positive_maps = []
        if len(set(text_prompts)) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            tokenized, caption_string, tokens_positive, _ = \
                self.get_tokens_and_prompts(
                    text_prompts[0], True)
            new_text_prompts = [caption_string] * len(batch_inputs)
            for gt_label in gt_labels:
                new_tokens_positive = [
                    tokens_positive[label] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
        else:
            for text_prompt, gt_label in zip(text_prompts, gt_labels):
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompt, True)
                new_tokens_positive = [
                    tokens_positive[label] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
                new_text_prompts.append(caption_string)

        language_dict_features = self.language_model(new_text_prompts)
        for i, data_samples in enumerate(batch_data_samples):
            # .bool().float() is very important
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            data_samples.gt_instances.positive_maps = positive_map

        visual_features = self.extract_feat(batch_inputs)

        losses = self.bbox_head.loss(visual_features, language_dict_features,
                                     batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - label_names (List[str]): Label names of bboxes.
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False

        if len(set(text_prompts)) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompts[0],
                                                     custom_entities)
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities)
                for text_prompt in text_prompts
            ]

        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        language_dict_features = self.language_model(list(text_prompts))

        for i, data_samples in enumerate(batch_data_samples):
            data_samples.token_positive_map = token_positive_maps[i]

        visual_features = self.extract_feat(batch_inputs)

        results_list = self.bbox_head.predict(
            visual_features,
            language_dict_features,
            batch_data_samples,
            rescale=rescale)

        for data_sample, pred_instances, entity in zip(batch_data_samples,
                                                       results_list, entities):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
