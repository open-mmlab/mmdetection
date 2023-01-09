import os.path as osp

try:
    import neptune.new as neptune
except ImportError:
    raise ImportError("Neptune client library not installed."
                      "Please refer to the installation guide:"
                      "https://docs.neptune.ai/setup/installation/")

import mmcv.runner.hooks as mmvch

from mmcv import Config
from mmcv.runner import IterBasedRunner, EpochBasedRunner
from mmcv.runner import HOOKS
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

    def __init__(
            self, *,
            interval: int = 20,
            base_namespace: str = "training",
            log_model: bool = False,
            log_checkpoint: bool = False,
            log_model_diagram: bool = False,
            num_eval_predictions: int = 50,
            **neptune_run_kwargs
    ):
        super().__init__()

        self.run = neptune.init_run(**neptune_run_kwargs)
        self.base_namespace = base_namespace
        self.base_handler = self.run[base_namespace]

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
        # Inspect CheckpointHook and EvalHook
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook

        # Save and Log config.
        if runner.meta is not None and runner.meta.get("exp_name", None) is not None:
            src_cfg_path = osp.join(runner.work_dir, runner.meta.get("exp_name", None))
            config = Config.fromfile(src_cfg_path)
            if osp.exists(src_cfg_path):
                self.base_handler["config"] = config.pretty_text
        else:
            runner.logger.warning("No meta information found in the runner. ")

        # Inspect CheckpointHook and EvalHook
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
            if isinstance(hook, (EvalHook, DistEvalHook)):
                self.eval_hook = hook
                self.val_dataset = self.eval_hook.dataloader.dataset

    def _log_buffer(self, runner, category, log_eval=True):
        assert category in ['epoch', 'iter']
        # only record lr of the first param group
        cur_lr = runner.current_lr()
        self.base_handler["train/" + category + "/" + 'learning_rate'].log(cur_lr)

        for key, value in runner.log_buffer.val_history.items():
            self.base_handler["train/" + category + "/" + key].log(value[-1])

        if not log_eval:
            return
        results = self.eval_hook.latest_results
        eval_results = self.val_dataset.evaluate(
            results, logger='silent', **self.eval_hook.eval_kwargs)

        for key, value in eval_results.items():
            self.base_handler["val/" + category + "/" + key].log(value)

    @master_only
    def after_train_iter(self, runner):
        print("ITER")
        # if self.by_epoch:
        #     return
        if not isinstance(runner, IterBasedRunner):
            return

        log_eval = self.every_n_iters(runner, self.eval_hook.interval)
        self._log_buffer(runner, 'iter', log_eval)

    @master_only
    def after_train_epoch(self, runner):
        print("EPOCH")
        # If runner has no Notion of Epoch.
        # Eg. IterBasedRunner
        # if not self.by_epoch:
        #     return
        if not isinstance(runner, EpochBasedRunner):
            return
        log_eval = self.every_n_epochs(runner, self.eval_hook.interval)
        self._log_buffer(runner, 'epoch', log_eval)

    @master_only
    def after_run(self, runner):
        if self.ckpt_hook is not None:
            out_path = self.ckpt_hook.out_dir
            self.base_handler['model/checkpoint/latest.pth'].upload(osp.join(out_path, 'latest.pth'))

        self.run.sync()
        self.run.stop()
