import torch.nn as nn
from functools import partial
from typing import Union, Sequence
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl
)

def layer_auto_wrap_policy(
        module,
        recurse: bool,
        nonwrapped_numel: int,
        layer_cls: Union[nn.Module, Sequence[nn.Module]],
) -> bool:
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for the leaf node or reminder
        return isinstance(module, tuple(layer_cls))


def checkpoint_check_fn(layer_cls: Union[nn.Module, Sequence[nn.Module]]):
    return lambda submodule: isinstance(submodule, tuple(layer_cls))

# non_reentrant_wrapper = partial(
#     checkpoint_wrapper,
#     offload_to_cpu=False,
#     checkpoint_impl=CheckpointImpl.NO_REENTRANT,
# )

