# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Sequence

import torch
from mmengine.model import BaseModel
from torch import nn

try:
    from transformers import AutoTokenizer, BertConfig
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

from mmdet.registry import MODELS


def generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_tokens_list):
    """Generate attention mask between each pair of special tokens.

    Only token pairs in between two special tokens are attended to
    and thus the attention mask for these pairs is positive.

    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.

    Returns:
        Tuple(Tensor, Tensor):
        - attention_mask is the attention mask between each tokens.
          Only token pairs in between two special tokens are positive.
          Shape: [bs, num_token, num_token].
        - position_ids is the position id of tokens within each valid sentence.
          The id starts from 0 whenenver a special token is encountered.
          Shape: [bs, num_token]
    """
    input_ids = tokenized['input_ids']
    bs, num_token = input_ids.shape
    # special_tokens_mask:
    # bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token),
                                      device=input_ids.device).bool()

    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token,
                  device=input_ids.device).bool().unsqueeze(0).repeat(
                      bs, 1, 1))
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1:col + 1,
                           previous_col + 1:col + 1] = True
            position_ids[row, previous_col + 1:col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device)
        previous_col = col

    return attention_mask, position_ids.to(torch.long)


@MODELS.register_module()
class BertModel(BaseModel):
    """BERT model for language embedding only encoder.

    Args:
        name (str, optional): name of the pretrained BERT model from
            HuggingFace. Defaults to bert-base-uncased.
        max_tokens (int, optional): maximum number of tokens to be
            used for BERT. Defaults to 256.
        pad_to_max (bool, optional): whether to pad the tokens to max_tokens.
             Defaults to True.
        use_sub_sentence_represent (bool, optional): whether to use sub
            sentence represent introduced in `Grounding DINO
            <https://arxiv.org/abs/2303.05499>`. Defaults to False.
        special_tokens_list (list, optional): special tokens used to split
            subsentence. It cannot be None when `use_sub_sentence_represent`
            is True. Defaults to None.
        add_pooling_layer (bool, optional): whether to adding pooling
            layer in bert encoder. Defaults to False.
        num_layers_of_embedded (int, optional): number of layers of
            the embedded model. Defaults to 1.
        use_checkpoint (bool, optional): whether to use gradient checkpointing.
             Defaults to False.
    """

    def __init__(self,
                 name: str = 'bert-base-uncased',
                 max_tokens: int = 256,
                 pad_to_max: bool = True,
                 use_sub_sentence_represent: bool = False,
                 special_tokens_list: list = None,
                 add_pooling_layer: bool = False,
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = False,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max

        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.language_backbone = nn.Sequential(
            OrderedDict([('body',
                          BertEncoder(
                              name,
                              add_pooling_layer=add_pooling_layer,
                              num_layers_of_embedded=num_layers_of_embedded,
                              use_checkpoint=use_checkpoint))]))

        self.use_sub_sentence_represent = use_sub_sentence_represent
        if self.use_sub_sentence_represent:
            assert special_tokens_list is not None, \
                'special_tokens should not be None \
                    if use_sub_sentence_represent is True'

            self.special_tokens = self.tokenizer.convert_tokens_to_ids(
                special_tokens_list)

    def forward(self, captions: Sequence[str], **kwargs) -> dict:
        """Forward function."""
        device = next(self.language_backbone.parameters()).device
        tokenized = self.tokenizer.batch_encode_plus(
            captions,
            max_length=self.max_tokens,
            padding='max_length' if self.pad_to_max else 'longest',
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True).to(device)
        input_ids = tokenized.input_ids
        if self.use_sub_sentence_represent:
            attention_mask, position_ids = \
                generate_masks_with_special_tokens_and_transfer_map(
                    tokenized, self.special_tokens)
            token_type_ids = tokenized['token_type_ids']

        else:
            attention_mask = tokenized.attention_mask
            position_ids = None
            token_type_ids = None

        tokenizer_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids
        }
        language_dict_features = self.language_backbone(tokenizer_input)
        if self.use_sub_sentence_represent:
            language_dict_features['position_ids'] = position_ids
            language_dict_features[
                'text_token_mask'] = tokenized.attention_mask.bool()
        return language_dict_features


class BertEncoder(nn.Module):
    """BERT encoder for language embedding.

    Args:
        name (str): name of the pretrained BERT model from HuggingFace.
                Defaults to bert-base-uncased.
        add_pooling_layer (bool): whether to add a pooling layer.
        num_layers_of_embedded (int): number of layers of the embedded model.
                Defaults to 1.
        use_checkpoint (bool): whether to use gradient checkpointing.
                Defaults to False.
    """

    def __init__(self,
                 name: str,
                 add_pooling_layer: bool = False,
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = False):
        super().__init__()
        if BertConfig is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')
        config = BertConfig.from_pretrained(name)
        config.gradient_checkpointing = use_checkpoint
        # only encoder
        self.model = HFBertModel.from_pretrained(
            name, add_pooling_layer=add_pooling_layer, config=config)
        self.language_dim = config.hidden_size
        self.num_layers_of_embedded = num_layers_of_embedded

    def forward(self, x) -> dict:
        mask = x['attention_mask']

        outputs = self.model(
            input_ids=x['input_ids'],
            attention_mask=mask,
            position_ids=x['position_ids'],
            token_type_ids=x['token_type_ids'],
            output_hidden_states=True,
        )

        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]
        features = torch.stack(encoded_layers[-self.num_layers_of_embedded:],
                               1).mean(1)
        # language embedding has shape [len(phrase), seq_len, language_dim]
        features = features / self.num_layers_of_embedded
        if mask.dim() == 2:
            embedded = features * mask.unsqueeze(-1).float()
        else:
            embedded = features

        results = {
            'embedded': embedded,
            'masks': mask,
            'hidden': encoded_layers[-1]
        }
        return results
