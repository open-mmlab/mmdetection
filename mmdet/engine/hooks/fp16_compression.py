# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class Fp16Compresssion(Hook):
    """Support ddp fp16 compression."""

    def before_run(self, runner):
        if runner.cfg.get('model_wrapper_cfg') is None:
            from torch.distributed.algorithms.ddp_comm_hooks import \
                default as comm_hooks
            model = runner.model
            model.register_comm_hook(
                state=None, hook=comm_hooks.fp16_compress_hook)
            runner.logger.info('use ddp fp16 compression')
