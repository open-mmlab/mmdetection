# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn as nn


class CLIPTextEncoder(nn.Module):

    def __init__(self, model_name='ViT-B/32'):
        super().__init__()
        import clip
        from clip.simple_tokenizer import SimpleTokenizer
        self.tokenizer = SimpleTokenizer()
        pretrained_model, _ = clip.load(model_name, device='cpu')
        self.clip = pretrained_model

    @property
    def device(self):
        return self.clip.device

    @property
    def dtype(self):
        return self.clip.dtype

    def tokenize(self,
                 texts: Union[str, List[str]],
                 context_length: int = 77) -> torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token]
                      for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                st = torch.randint(len(tokens) - context_length + 1,
                                   (1, ))[0].item()
                tokens = tokens[st:st + context_length]
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def forward(self, text):
        text = self.tokenize(text)
        text_features = self.clip.encode_text(text)
        return text_features
