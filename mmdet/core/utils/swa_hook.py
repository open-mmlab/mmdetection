import os.path as osp

from mmcv.runner import HOOKS, Hook
from mmcv.runner.checkpoint import save_checkpoint
from torch.optim.swa_utils import AveragedModel

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
            if not isinstance(eval_hook, EvalHook) or \
                   isinstance(eval_hook, DistEvalHook):
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
            self.log_buffer = runner.log_buffer
            self.meta['hook_msgs']['last_ckpt'] = filename
            self.eval_hook.after_train_epoch(self)

    def after_run(self, runner):
        # since BN layers in the backbone are frozen,
        # we do not need to update the BN for the swa model
        pass

    def before_epoch(self, runner):
        pass
