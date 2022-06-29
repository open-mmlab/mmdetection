# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class ModelSwitchHook(Hook):
    """Model Switch Hook.

    Args:
        interval (int): Update teacher's parameter every interval iteration.
            Defaults to 1.
    """

    def __init__(self, interval: int = 1) -> None:
        self.interval = interval

    def before_run(self, runner):
        """To check that teacher model and student model exist."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        assert hasattr(model, 'teacher')
        assert hasattr(model, 'student')

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Update teacher's parameter every self.interval iterations."""
        if (runner.iter + 1) % self.interval != 0:
            return
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        for (src_name, src_param), (dst_name, dst_param) \
                in zip(model.student.named_parameters(),
                       model.teacher.named_parameters()):
            tmp = dst_param.data.clone()
            dst_param.data.copy_(src_param.data)
            src_param.data.copy_(tmp)
