# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class Fp16CompresssionHook(Hook):
    """Support fp16 compression in DDP mode.

    In detectron2, vitdet use Fp16CompresssionHook in training process
    Fp16CompresssionHook can reduce training time and improve bbox mAP when you
    use Fp16CompresssionHook, training time reduce form 3 days to 2 days and
    box mAP from 51.4 to 51.6
    """

    def before_train(self, runner):

        if runner.distributed:
            if runner.cfg.get('model_wrapper_cfg') is None:
                from torch.distributed.algorithms.ddp_comm_hooks import \
                    default as comm_hooks
                runner.model.register_comm_hook(
                    state=None, hook=comm_hooks.fp16_compress_hook)
                runner.logger.info('use fp16 compression in DDP mode')
