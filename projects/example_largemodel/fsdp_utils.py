from typing import Sequence, Union

import torch.nn as nn

from mmdet.models.backbones.swin import SwinBlock
from mmdet.models.layers.transformer.deformable_detr_layers import \
    DeformableDetrTransformerEncoderLayer


# TODO: The new version of configs does not support passing a module list,
#  so for now, it can only be hard-coded. We will fix this issue in the future.
def layer_auto_wrap_policy(
    module,
    recurse: bool,
    nonwrapped_numel: int,
    layer_cls: Union[nn.Module, Sequence[nn.Module]] = (
        SwinBlock, DeformableDetrTransformerEncoderLayer),
) -> bool:
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for
        # the leaf node or reminder
        return isinstance(module, tuple(layer_cls))


def checkpoint_check_fn(submodule,
                        layer_cls: Union[nn.Module, Sequence[nn.Module]] = (
                            SwinBlock, DeformableDetrTransformerEncoderLayer)):
    return isinstance(submodule, tuple(layer_cls))


# non_reentrant_wrapper = partial(
#     checkpoint_wrapper,
#     offload_to_cpu=False,
#     checkpoint_impl=CheckpointImpl.NO_REENTRANT,
# )
