from mmdet.models.backbones.swin import SwinBlock
from mmdet.models.layers.transformer.deformable_detr_layers import DeformableDetrTransformerEncoderLayer
from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

def name_auto_wrap_policy(
    module,
    recurse: bool,
    nonwrapped_numel: int,
    layer_cls,
) -> bool:
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for the leaf node or reminder
        return isinstance(module, tuple([SwinBlock,DeformableDetrTransformerEncoderLayer]))

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    offload_to_cpu=False,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn = lambda submodule: isinstance(submodule, tuple([SwinBlock, DeformableDetrTransformerEncoderLayer]))

def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fdsp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
