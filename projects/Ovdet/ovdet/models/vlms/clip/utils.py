from typing import List, Union

import torch

from .simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def tokenize(
    texts: Union[str, List[str]],
    context_length: int = 77,
    truncate: bool = False,
) -> torch.LongTensor:
    """Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The ovd length to use; all CLIP models use 77 as the ovd length

    truncate: bool
        Whether to truncate the text in case its encoding is longer
        than the ovd length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens,
    shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder['<|startoftext|>']
    eot_token = _tokenizer.encoder['<|endoftext|>']
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f'Input {texts[i]} is too long '
                                   f'for ovd length {context_length}')
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def tokenize_dynamic(texts, context_length: int = 77, truncate: bool = False):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder['<|startoftext|>']
    eot_token = _tokenizer.encoder['<|endoftext|>']
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    lengths = [len(tokens) for tokens in all_tokens]
    context_length = min(context_length, max(lengths))
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f'Input {texts[i]} is too long '
                                   f'for ovd length {context_length}')
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
