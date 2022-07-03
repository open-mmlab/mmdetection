# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class ModelSwapHook(Hook):
    """Model Swap Hook.

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
        """Swap the parameter of model between teacher and student."""
        if (runner.iter + 1) % self.interval != 0:
            return
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        teacher_param = itertools.chain(model.teacher.parameters(),
                                        model.teacher.buffers())
        student_param = itertools.chain(model.student.parameters(),
                                        model.student.buffers())
        for tp, sp in zip(teacher_param, student_param):
            tmp = tp.data.clone().detach()
            tp.data.copy_(sp.data)
            sp.data.copy_(tmp)
