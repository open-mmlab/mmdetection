# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class MeanTeacher(Hook):
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
        skip_buffers (bool): Whether to skip the model buffers, such as
            batchnorm running stats (running_mean, running_var), it does not
            perform the ema operation. Default to False.
        momentum_fun (func, optional): The function to change momentum
            during early iteration (also warmup) to help early training.
            It uses `momentum` as a constant. Defaults to None.
    """

    def __init__(self,
                 momentum=0.001,
                 interval=1,
                 skip_buffer=False,
                 momentum_fun=None):
        assert 0 < momentum < 1
        self.momentum = momentum
        self.interval = interval
        self.skip_buffers = skip_buffer
        self.momentum_fun = momentum_fun

    def before_run(self, runner):
        """To check that teacher model and student model exist."""
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, 'teacher')
        assert hasattr(model, 'student')
        # only do it at initial stage
        if runner.iter == 0:
            self.momentum_update(model, 1)

    def get_momentum(self, runner):
        return self.momentum_fun(runner.iter) if self.momentum_fun else \
            self.momentum

    def after_train_iter(self, runner):
        """Update teacher's parameter every self.interval iterations."""
        if (runner.iter + 1) % self.interval != 0:
            return
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        momentum = self.get_momentum(runner)
        runner.log_buffer.output['ema_momentum'] = momentum
        self.momentum_update(model, momentum)

    def momentum_update(self, model, momentum):
        if self.skip_buffers:
            for (src_name, src_parm), (dst_name, dst_parm) in zip(
                    model.student.named_parameters(),
                    model.teacher.named_parameters()):
                dst_parm.data.mul_(1 - momentum).add_(
                    src_parm.data, alpha=momentum)
        else:
            for (src_parm,
                 dst_parm) in zip(model.student.state_dict().values(),
                                  model.teacher.state_dict().values()):
                # exclude num_tracking
                if dst_parm.dtype.is_floating_point:
                    dst_parm.data.mul_(1 - momentum).add_(
                        src_parm.data, alpha=momentum)
