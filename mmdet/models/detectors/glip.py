# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from .single_stage import SingleStageDetector
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
import nltk
import re
import torch
import copy


def find_noun_phrases(caption: str):
    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = list()
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    return noun_phrases


def remove_punctuation(text: str) -> str:
    punct = ['|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^',
             '\'', '\"', '’', '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
             ]
    for p in punct:
        text = text.replace(p, '')
    return text.strip()


def run_ner(caption):
    # There is two cat and a remote in the picture
    # 离线的 NER 算法： ['cat', 'a remote', 'the picture']
    noun_phrases = find_noun_phrases(caption)
    # 移除标点符号
    noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
    # 最终的实体
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
        except:
            print("noun entities:", noun_phrases)
            print("entity:", entity)
            print("caption:", caption.lower())
    # [[[13, 16]], [[21, 29]], [[33, 44]]]
    # 表示一共有 3 个实体，第一个实体 cat 位于输入句子的 [13:16] 位置，其他类推
    return tokens_positive, noun_phrases


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    # 256 意思是输入的任何一个命名实体不能超过 256 token，这个应该不会存在吧
    # 注意，这里的第一个维度不是句子个数，而是句子中的实体个数
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)  # (3, 256)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                # There is two cat and a remote in the picture
                # token 长度是 12
                # 假设这个 beg end = 13 16 其实际上是对应 cat 在句子中的位置
                # char_to_token 可以对应的找到其在 tokenized 的对应偏移位置即 4 和 4 = tokenized[4:4]
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                print("beg:", beg, "end:", end)
                print("token_positive:", tokens_positive)
                # print("beg_pos:", beg_pos, "end_pos:", end_pos)
                raise e
            if beg_pos is None:  # 啥时候会是 None？
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            # 对应的位置值设置为 1，相当于这些位置是命名实体位置
            positive_map[j, beg_pos: end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)  # 每个实体 token 归一化，确保每个命名实体 token 响应和为 1


def create_positive_map_label_to_token_from_positive_map(positive_map, plus=0):
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token  # {1: [4], 2: [6, 7], 3: [9, 10]}


@MODELS.register_module()
class GLIP(SingleStageDetector):

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

        self._text_prompts = None
        self._positive_maps = None
        self._language_dict_features = None
        self._entities = None

    def get_tokens_positive(self, original_caption, custom_entities):
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                if not original_caption.endswith('.'):
                    original_caption = original_caption + ' . '
                original_caption = original_caption.split(' . ')
                original_caption = list(filter(lambda x: len(x) > 0, original_caption))
            # 如果是类别列表，则直接拼接，中间用 ' . ' 区分
            caption_string = ""
            tokens_positive = []
            seperation_tokens = " . "
            for word in original_caption:
                # 由于一个类别就是一个实体，因此可以直接得到的 positive，无需进行命名实体识别。方便后续还原类别
                tokens_positive.append([[len(caption_string), len(caption_string) + len(word)]])
                caption_string += word
                caption_string += seperation_tokens
            tokenized = self.language_model.tokenizer([caption_string], return_tensors="pt")
            self._entities = original_caption
        else:
            # 假设输入是 There is two cat and a remote in the picture
            # 将输入文本 token 化，会加入开始和结束符
            # 编码后长度为 12
            if not original_caption.endswith('.'):
                original_caption = original_caption + ' . '

            tokenized = self.language_model.tokenizer([original_caption], return_tensors="pt")
            # 识别文本中的名词，作为类别，并计算对应名词 token 位置
            # 假设输入是 There is two cat and a remote in the picture
            # 实体： ['cat', 'a remote', 'the picture']
            # tokens_positive=[[[13, 16]], [[21, 29]], [[33, 44]]]
            # 表示一共有 3 个实体，第一个实体 cat 位于输入句子的 [13:16] 位置，其他类推
            tokens_positive, noun_phrases = run_ner(original_caption)  # 找到句子中的命名实体
            self._entities = noun_phrases
            caption_string = original_caption

        positive_map = create_positive_map(tokenized, tokens_positive)
        positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=1)
        return positive_map_label_to_token, caption_string

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                custom_entities=False,
                rescale: bool = True) -> SampleList:
        text_prompts = [data_samples.caption for data_samples in batch_data_samples]
        if text_prompts != self._text_prompts:
            # avoid redundant computation
            self._text_prompts = text_prompts
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same, so there is no need to calculate them multiple times.
                _positive_maps_and_prompts = [self.get_tokens_positive(text_prompts[0], custom_entities)] * len(
                    batch_inputs)
            else:
                _positive_maps_and_prompts = [self.get_tokens_positive(text_prompt, custom_entities) for text_prompt in
                                              text_prompts]

            self._positive_maps, text_prompts = zip(*_positive_maps_and_prompts)
            self._language_dict_features = self.language_model(text_prompts)

        for i, data_samples in enumerate(batch_data_samples):
            data_samples.token_positive_map = self._positive_maps[i]

        visual_features = self.extract_feat(batch_inputs)

        results_list = self.bbox_head.predict(
            visual_features, copy.deepcopy(self._language_dict_features), batch_data_samples, rescale=rescale)

        for data_sample, pred_instances in zip(batch_data_samples, results_list):
            if len(pred_instances) > 0:
                # Important
                pred_instances.labels = pred_instances.labels-1

                label_names = []
                for labels in pred_instances.labels:
                    if labels >= len(self._entities):
                        warnings.warn('The unexpected output indicates an issue with '
                                      'named entity recognition. You can try '
                                      'setting custom_entities=True and running '
                                      'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(self._entities[labels])
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
