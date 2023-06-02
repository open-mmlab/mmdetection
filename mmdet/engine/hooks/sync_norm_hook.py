# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

from mmengine.dist import get_dist_info
from mmengine.hooks import Hook
from torch import nn

from mmdet.registry import HOOKS
from mmdet.utils import all_reduce_dict


def get_norm_states(module: nn.Module) -> OrderedDict:
    """Get the state_dict of batch norms in the module."""
    async_norm_states = OrderedDict()
    for name, child in module.named_modules():
        if isinstance(child, nn.modules.batchnorm._NormBase):
            for k, v in child.state_dict().items():
                async_norm_states['.'.join([name, k])] = v
    return async_norm_states


@HOOKS.register_module()
class SyncNormHook(Hook):
    """Synchronize Norm states before validation, currently used in YOLOX."""

    def before_val_epoch(self, runner):
        """Synchronizing norm."""
        module = runner.model
        _, world_size = get_dist_info()
        if world_size == 1:
            return
        norm_states = get_norm_states(module)
        if len(norm_states) == 0:
            return
        # TODO: use `all_reduce_dict` in mmengine
        norm_states = all_reduce_dict(norm_states, op='mean')
        module.load_state_dict(norm_states, strict=False)
