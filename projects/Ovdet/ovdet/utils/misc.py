import json
from functools import partial

import torch
from six.moves import map, zip


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def load_class_freq(path='data/metadata/lvis_v1_train_cat_norare_info.json',
                    freq_weight=1.0,
                    min_count=0):
    cat_info = json.load(open(path, 'r'))
    cat_info = torch.tensor([
        max(c['image_count'], min_count)
        for c in sorted(cat_info, key=lambda x: x['id'])
    ])
    freq_weight = cat_info.float()**freq_weight
    return freq_weight
