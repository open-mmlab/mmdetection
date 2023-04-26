from mmdet.registry import MODELS
from mmengine.model import BaseModel
from transformers import AutoTokenizer

from copy import deepcopy
import numpy as np
import torch
from torch import nn

from transformers import BertConfig
from transformers import BertModel as TBertModel
from collections import OrderedDict
from typing import Sequence


class BertEncoder(nn.Module):
    def __init__(self, name='bert-base-uncased', num_layers=1, use_checkpoint=False):
        super(BertEncoder, self).__init__()
        config = BertConfig.from_pretrained(name)
        config.gradient_checkpointing = use_checkpoint
        self.model = TBertModel.from_pretrained(name, add_pooling_layer=False, config=config)
        self.language_dim = 768
        self.num_layers = num_layers

    def forward(self, x):
        input = x["input_ids"]
        mask = x["attention_mask"]

        # with padding, always 256
        outputs = self.model(
            input_ids=input,
            attention_mask=mask,
            output_hidden_states=True,
        )
        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]
        features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)

        # language embedding has shape [len(phrase), seq_len, language_dim]
        features = features / self.num_layers

        embedded = features * mask.unsqueeze(-1).float()
        aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        ret = {
            "aggregate": aggregate,
            "embedded": embedded,
            "masks": mask,
            "hidden": encoded_layers[-1]
        }
        return ret


@MODELS.register_module()
class BertModel(BaseModel):
    def __init__(self,
                 max_tokens=256,
                 pad_to_max=True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.language_backbone = nn.Sequential(OrderedDict([("body", BertEncoder('bert-base-uncased'))]))

    def forward(self,
                captions: Sequence[str],
                data_samples=None,
                mode: str = 'tensor'):
        device = next(self.language_backbone.parameters()).device
        tokenized = self.tokenizer.batch_encode_plus(captions,
                                                     max_length=self.max_tokens,
                                                     padding='max_length' if self.pad_to_max else "longest",
                                                     return_special_tokens_mask=True,
                                                     return_tensors='pt',
                                                     truncation=True).to(device)  # 想并行计算

        tokenizer_input = {"input_ids": tokenized.input_ids,
                           "attention_mask": tokenized.attention_mask}
        language_dict_features = self.language_backbone(tokenizer_input)
        return language_dict_features
