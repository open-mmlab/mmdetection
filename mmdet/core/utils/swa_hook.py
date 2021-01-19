import os.path as osp
from copy import deepcopy

import torch
from mmcv.runner import HOOKS, Hook
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.log_buffer import LogBuffer

from mmdet.core import DistEvalHook, EvalHook


@HOOKS.register_module()
class SWAHook(Hook):
    r"""SWA Object Detection Hook.

        This hook works together with SWA training config files to train
        SWA object detectors <https://arxiv.org/abs/2012.12645>.

        Args:
            swa_eval (bool): Whether to evaluate the swa model.
                Defaults to True.
            eval_hook (Hook): Hook class that contains evaluation functions.
                Defaults to None.
    """

    def __init__(self, swa_eval=True, eval_hook=None):
        if not isinstance(swa_eval, bool):
            raise TypeError('swa_eval must be a bool, but got'
                            f'{type(swa_eval)}')
        if swa_eval:
            if not isinstance(eval_hook, EvalHook) and \
               not isinstance(eval_hook, DistEvalHook):
                raise TypeError('eval_hook must be either a EvalHook or a '
                                'DistEvalHook when swa_eval = True, but got'
                                f'{type(eval_hook)}')
        self.swa_eval = swa_eval
        self.eval_hook = eval_hook

    def before_run(self, runner):
        """Construct the averaged model which will keep track of the running
        averages of the parameters of the model."""
        model = runner.model
        self.model = AveragedModel(model)

        self.log_buffer = LogBuffer()
        self.meta = runner.meta
        if self.meta is None:
            self.meta = dict()
            self.meta.setdefault('hook_msgs', dict())

    def after_train_epoch(self, runner):
        """Update the parameters of the averaged model, save and evaluate the
        updated averaged model."""
        model = runner.model
        # update the parameters of the averaged model
        self.model.update_parameters(model)

        # save the swa model
        runner.logger.info(
            f'Saving swa model at swa-training {runner.epoch + 1} epoch')
        filename = 'swa_model_{}.pth'.format(runner.epoch + 1)
        filepath = osp.join(runner.work_dir, filename)
        optimizer = runner.optimizer
        self.meta['hook_msgs']['last_ckpt'] = filepath
        save_checkpoint(
            self.model.module, filepath, optimizer=optimizer, meta=self.meta)

        # evaluate the swa model
        if self.swa_eval:
            self.work_dir = runner.work_dir
            self.rank = runner.rank
            self.epoch = runner.epoch
            self.logger = runner.logger
            self.meta['hook_msgs']['last_ckpt'] = filename
            self.eval_hook.after_train_epoch(self)
            for name, val in self.log_buffer.output.items():
                name = 'swa_' + name
                runner.log_buffer.output[name] = val
            runner.log_buffer.ready = True
            self.log_buffer.clear()

    def after_run(self, runner):
        # since BN layers in the backbone are frozen,
        # we do not need to update the BN for the swa model
        pass

    def before_epoch(self, runner):
        pass


class AveragedModel(torch.nn.Module):
    r"""Implements averaged model for Stochastic Weight Averaging (SWA).

    AveragedModel class creates a copy of the provided model on the device
    and allows to compute running averages of the parameters of the model.

    Args:
        model (torch.nn.Module): model to use with SWA
        device (torch.device, optional): if provided, the averaged model
            will be stored on the device. Defaults to None.
        avg_fn (function, optional): the averaging function used to update
            parameters; the function must take in the current value of the
            AveragedModel parameter, the current value of model
            parameter and the number of models already averaged; if None,
            equally weighted average is used. Defaults to None.
    """

    def __init__(self, model, device=None, avg_fn=None):
        super(AveragedModel, self).__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:

            def avg_fn(averaged_model_parameter, model_parameter,
                       num_averaged):
                return averaged_model_parameter + (
                    model_parameter - averaged_model_parameter) / (
                        num_averaged + 1)

        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(
                    self.avg_fn(p_swa.detach(), p_model_,
                                self.n_averaged.to(device)))
        self.n_averaged += 1
