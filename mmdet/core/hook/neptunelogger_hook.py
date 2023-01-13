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

from mmdet import version
from mmdet.core import DistEvalHook, EvalHook


@HOOKS.register_module()
class NeptuneHook(mmvch.logger.neptune.NeptuneLoggerHook):
    """Log metadata to Neptune.

    This hook will automatically log training or evaluation metadata
    to Neptune.ai.
    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    INTEGRATION_VERSION_KEY = 'source_code/integrations/neptune-mmdetection'

    def __init__(self,
                 *,
                 api_token: str = None,
                 project: str = None,
                 interval: int = 50,
                 base_namespace: str = 'training',
                 log_model: bool = False,
                 log_checkpoint: bool = False,
                 log_model_diagram: bool = False,
                 num_eval_predictions: int = 50,
                 **kwargs) -> None:
        super().__init__()

        self._run = neptune.init_run(api_token=api_token, project=project)
        self.base_namespace = base_namespace
        self.base_handler = self._run[base_namespace]

        self.interval = interval

        self.log_model = log_model
        self.log_checkpoint = log_checkpoint
        self.log_model_diagram = log_model_diagram

        self.num_eval_predictions = num_eval_predictions
        self.log_eval_predictions = (num_eval_predictions > 0)

        self.ckpt_hook: CheckpointHook = None
        self.ckpt_interval: int = None

        self.eval_hook: EvalHook = None
        self.eval_interval: int = None

        self.val_dataset = None

        self.kwargs = kwargs

    def _log_integration_version(self) -> None:
        self._run[self.INTEGRATION_VERSION_KEY] = version.__version__

    def _log_config(self, runner) -> None:
        if runner.meta is not None and runner.meta.get('exp_name',
                                                       None) is not None:
            src_cfg_path = osp.join(runner.work_dir,
                                    runner.meta.get('exp_name', None))
            config = Config.fromfile(src_cfg_path)
            if osp.exists(src_cfg_path):
                self.base_handler['config'] = config.pretty_text
        else:
            runner.logger.warning('No meta information found in the runner. ')

    @master_only
    def before_run(self, runner) -> None:
        """Logs config if exists, inspects the hooks in search of checkpointing
        and evaluation hooks.

        Raises a warning if checkpointing is enabled, but the dedicated hook is
        not present. Raises a warning if evaluation logging is enabled, but the
        dedicated hook is not present.
        """
        self._log_integration_version()
        self._log_config(runner)

        # Inspect CheckpointHook and EvalHook
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
            if isinstance(hook, (EvalHook, DistEvalHook)):
                self.eval_hook = hook

        if self.log_checkpoint:
            if self.ckpt_hook is None:
                self.log_checkpoint = False
                runner.logger.warning(
                    'WARNING ABOUT CHECKPOINT HOOK NOT PRESENT')
            else:
                self.ckpt_interval = self.ckpt_hook.interval

        if self.log_eval_predictions:
            if self.eval_hook is None:
                self.log_eval_predictions = False
                runner.logger.warning('WARNING ABOUT EVAL HOOK NOT PRESENT')
            else:
                self.eval_interval = self.eval_hook.interval
                self.val_dataset = self.eval_hook.dataloader.dataset

    def _log_buffer(self, runner, category, log_eval=True) -> None:
        assert category in ['epoch', 'iter']
        # only record lr of the first param group
        cur_lr = runner.current_lr()
        self.base_handler['train/' + category + '/' +
                          'learning_rate'].extend(cur_lr)

        for key, value in runner.log_buffer.val_history.items():
            self.base_handler['train/' + category + '/' + key].log(value[-1])

        if log_eval and self.eval_hook._should_evaluate(runner):

            results = self.eval_hook.latest_results

            eval_results = self.val_dataset.evaluate(
                results, logger='silent', **self.eval_hook.eval_kwargs)

            for key, value in eval_results.items():
                self.base_handler['val/' + category + '/' + key].log(value)

    def _log_checkpoint(self,
                        runner,
                        ext='pth',
                        final=False,
                        mode='epoch') -> None:
        assert mode in ['epoch', 'iter']

        if self.ckpt_hook is None:
            return

        if final:
            file_name = 'latest'
        else:
            file_name = f'{mode}_{getattr(runner, mode) + 1}'

        file_path = osp.join(self.ckpt_hook.out_dir, f'{file_name}.{ext}')

        neptune_checkpoint_path = osp.join('model/checkpoint/',
                                           f'{file_name}.{ext}')

        if not osp.exists(file_path):
            warnings.warn('WARNING ABOUT CHECKPOINT FILE NOT FOUND')
            return
        with open(file_path, 'rb') as fp:
            self._run[neptune_checkpoint_path] = File.from_stream(fp)

    @master_only
    def after_train_iter(self, runner) -> None:
        """For an iter-based runner logs evaluation metadata, as well as
        checkpoints (if enabled by the user)."""
        if not isinstance(runner, IterBasedRunner):
            return

        log_eval = self.every_n_iters(runner, self.eval_hook.interval)
        self._log_buffer(runner, 'iter', log_eval)
        if self._should_upload_checkpoint(runner):
            self._log_checkpoint(runner, final=False, mode='iter')

    def _should_upload_checkpoint(self, runner) -> bool:
        if isinstance(runner, EpochBasedRunner):
            return self.log_checkpoint and \
                (runner.epoch + 1) % self.ckpt_hook.interval == 0
        elif isinstance(runner, IterBasedRunner):
            return self.log_checkpoint and \
                (runner.iter + 1) % self.ckpt_hook.interval == 0

    @master_only
    def after_train_epoch(self, runner) -> None:
        """For an epoch-based runner logs evaluation metadata, as well as
        checkpoints (if enabled by the user)."""
        if not isinstance(runner, EpochBasedRunner):
            return
        log_eval = self.every_n_epochs(runner, self.eval_hook.interval)
        self._log_buffer(runner, 'epoch', log_eval)

        if self._should_upload_checkpoint(runner):
            self._log_checkpoint(runner, final=False)

    @master_only
    def after_run(self, runner) -> None:
        """If enabled by the user, logs final model checkpoint, syncs and stops
        the Neptune run."""
        if self.log_model:
            self._log_checkpoint(runner, final=True)

        runner.logger.info('SYNCING')
        self._run.sync()
        self._run.stop()
