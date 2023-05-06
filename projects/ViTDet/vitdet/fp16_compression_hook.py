# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class Fp16CompresssionHook(Hook):
    """Support fp16 compression in DDP mode."""

    def before_train(self, runner):

        if runner.distributed:
            if runner.cfg.get('model_wrapper_cfg') is None:
                from torch.distributed.algorithms.ddp_comm_hooks import \
                    default as comm_hooks
                runner.model.register_comm_hook(
                    state=None, hook=comm_hooks.fp16_compress_hook)
                runner.logger.info('use fp16 compression in DDP mode')
