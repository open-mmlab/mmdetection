# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class MeanTeacherHook(Hook):
    """Mean Teacher Hook.
    Mean Teacher is an efficient semi-supervised learning method.
    This method requires two models with exactly the same structure,
    as the student model and the teacher model, respectively.
    The student model updates the parameters through gradient descent,
    and the teacher model updates the parameters through
    exponential moving average of the student model.
    Compared with the student model, the teacher model
    is smoother and accumulates more knowledge.
    The paper link is https://arxiv.org/abs/1703.01780.
    Args:
        momentum (float): The momentum used for updating teacher's parameter.
            Teacher's parameter are updated with the formula:
           `teacher = (1-momentum) * teacher + momentum * student`.
            Defaults to 0.001.
        interval (int): Update teacher's parameter every interval iteration.
            Defaults to 1.
    """

    def __init__(self, momentum: float = 0.001, interval: int = 1) -> None:
        assert 0 < momentum < 1
        self.momentum = momentum
        self.interval = interval

    def before_run(self, runner):
        """To check that teacher model and student model exist."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        assert hasattr(model, 'teacher')
        assert hasattr(model, 'student')
        # only do it at initial stage
        if runner.iter == 0:
            self.momentum_update(model, 1)

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
        self.momentum_update(model, self.momentum)

    def momentum_update(self, model, momentum):
        for (src_name, src_param), (dst_name, dst_param) \
                in zip(model.student.named_parameters(),
                       model.teacher.named_parameters()):
            dst_param.data.mul_(1 - momentum).add_(
                src_param.data, alpha=momentum)
