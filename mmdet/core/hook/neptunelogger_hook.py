# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

try:
    import neptune.new as neptune
    from neptune.new.types import File
except ImportError:
    raise ImportError('Neptune client library not installed.'
                      'Please refer to the installation guide:'
                      'https://docs.neptune.ai/setup/installation/')

import mmcv.runner.hooks as mmvch
from mmcv import Config
from mmcv.runner import HOOKS, EpochBasedRunner, IterBasedRunner
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.checkpoint import CheckpointHook

from mmdet.core import DistEvalHook, EvalHook


@HOOKS.register_module()
class NeptuneHook(mmvch.logger.neptune.NeptuneLoggerHook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.
    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self,
                 *,
                 interval: int = 20,
                 base_namespace: str = 'training',
                 log_model: bool = False,
                 log_checkpoint: bool = False,
                 log_model_diagram: bool = False,
                 num_eval_predictions: int = 50,
                 **neptune_run_kwargs):
        super().__init__()

        self._run = neptune.init_run(**neptune_run_kwargs)
        self.base_namespace = base_namespace
        self.base_handler = self._run[base_namespace]

        self.interval = interval

        self.log_model = log_model
        self.log_checkpoint = log_checkpoint
        self.log_model_diagram = log_model_diagram

        self.num_eval_predictions = num_eval_predictions
        self.log_eval_predictions = (num_eval_predictions > 0)

        self.ckpt_hook: CheckpointHook
        self.eval_hook: EvalHook

    @master_only
    def before_run(self, runner):
        # Save and Log config.
        if runner.meta is not None and runner.meta.get('exp_name',
                                                       None) is not None:
            src_cfg_path = osp.join(runner.work_dir,
                                    runner.meta.get('exp_name', None))
            config = Config.fromfile(src_cfg_path)
            if osp.exists(src_cfg_path):
                self.base_handler['config'] = config.pretty_text
        else:
            runner.logger.warning('No meta information found in the runner. ')

        # Inspect CheckpointHook and EvalHook
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
            if isinstance(hook, (EvalHook, DistEvalHook)):
                self.eval_hook = hook
                self.val_dataset = self.eval_hook.dataloader.dataset

        if self.log_checkpoint and self.ckpt_hook is None:
            warnings.warn('WARNING ABOUT CHECKPOINTER NOT PRESENT')

    def _log_buffer(self, runner, category, log_eval=True):
        assert category in ['epoch', 'iter']
        # only record lr of the first param group
        cur_lr = runner.current_lr()
        self.base_handler['train/' + category + '/' +
                          'learning_rate'].log(cur_lr)

        for key, value in runner.log_buffer.val_history.items():
            self.base_handler['train/' + category + '/' + key].log(value[-1])

        if not log_eval:
            return
        results = self.eval_hook.latest_results
        eval_results = self.val_dataset.evaluate(
            results, logger='silent', **self.eval_hook.eval_kwargs)

        for key, value in eval_results.items():
            self.base_handler['val/' + category + '/' + key].log(value)

    def _log_checkpoint(self, runner, ext='pth', final=False):
        if self.ckpt_hook is None:
            return

        if final:
            file_name = 'latest'
        else:
            file_name = f'epoch_{runner.epoch}'

        file_path = osp.join(self.ckpt_hook.out_dir, f'{file_name}.{ext}')

        neptune_checkpoint_path = osp.join('model/checkpoint/',
                                           f'{file_name}.{ext}')

        if not osp.exists(file_path):
            warnings.warn('WARNING ABOUT CHECKPOINT FILE NOT FOUND')
            return
        with open(file_path, 'rb') as fp:
            self._run[neptune_checkpoint_path] = File.from_stream(fp)

    @master_only
    def after_train_iter(self, runner):
        if not isinstance(runner, IterBasedRunner):
            return

        log_eval = self.every_n_iters(runner, self.eval_hook.interval)
        self._log_buffer(runner, 'iter', log_eval)

    def _should_upload_checkpoint(self, runner) -> bool:
        if isinstance(runner, EpochBasedRunner):
            return self.log_checkpoint and runner.epoch != 0 and \
                runner.epoch % self.ckpt_hook.interval == 0

    @master_only
    def after_train_epoch(self, runner):
        if not isinstance(runner, EpochBasedRunner):
            return
        log_eval = self.every_n_epochs(runner, self.eval_hook.interval)
        self._log_buffer(runner, 'epoch', log_eval)

        if self._should_upload_checkpoint(runner):
            self._log_checkpoint(runner, final=False)

    @master_only
    def after_run(self, runner):
        if self.log_model:
            self._log_checkpoint(runner, final=True)

        print('SYNCING')
        self._run.sync()
        self._run.stop()
