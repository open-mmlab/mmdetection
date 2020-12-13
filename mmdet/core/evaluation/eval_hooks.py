import os.path as osp
import warnings
from math import inf

import mmcv
from mmcv.runner import Hook
from torch.utils.data import DataLoader

from mmdet.utils import get_root_logger


class EvalHook(Hook):
    """Evaluation hook.

    Notes:
        If new arguments are added for EvalHook, tools/test.py,
        tools/eval_metric.py may be effected.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        start (int, optional): Evaluation starting epoch. It enables evaluation
            before the training starts if ``start`` <= the resuming epoch.
            If None, whether to evaluate is merely decided by ``interval``.
            Default: None.
        interval (int): Evaluation interval (by epochs). Default: 1.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be save in best.json.
            Options are the evaluation metrics to the test dataset. e.g.,
            ``bbox_mAP``, ``segm_mAP`` for bbox detection and instance
            segmentation. ``AR@100`` for proposal recall. If ``save_best`` is
            ``auto``, the first key will be used. The interval of
            ``CheckpointHook`` should device EvalHook. Default: None.
        rule (str, optional): Comparison rule for best score. If set to None,
            it will infer a reasonable rule. Keys such as 'mAP' or 'AR' will
            be inferred by 'greater' rule. Keys contain 'loss' will be inferred
             by 'less' rule. Options are 'greater', 'less'. Default: None.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    greater_keys = ['mAP', 'AR']
    less_keys = ['loss']

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 save_best=None,
                 rule=None,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        if not interval > 0:
            raise ValueError(f'interval must be positive, but got {interval}')
        if start is not None and start < 0:
            warnings.warn(
                f'The evaluation start epoch {start} is smaller than 0, '
                f'use 0 instead', UserWarning)
            start = 0
        self.dataloader = dataloader
        self.interval = interval
        self.start = start
        assert isinstance(save_best, str) or save_best is None
        self.save_best = save_best
        self.eval_kwargs = eval_kwargs
        self.initial_epoch_flag = True

        self.logger = get_root_logger()

        if self.save_best is not None:
            self._init_rule(rule, self.save_best)

    def _init_rule(self, rule, key_indicator):
        """Initialize rule, key_indicator, comparison_func, and best score.

        Args:
            rule (str | None): Comparison rule for best score.
            key_indicator (str | None): Key indicator to determine the
                comparison rule.
        """
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f'rule must be greater, less or None, '
                           f'but got {rule}.')

        if rule is None:
            if key_indicator != 'auto':
                if any(key in key_indicator for key in self.greater_keys):
                    rule = 'greater'
                elif any(key in key_indicator for key in self.less_keys):
                    rule = 'less'
                else:
                    raise ValueError(f'Cannot infer the rule for key '
                                     f'{key_indicator}, thus a specific rule '
                                     f'must be specified.')
        self.rule = rule
        self.key_indicator = key_indicator
        if self.rule is not None:
            self.compare_func = self.rule_map[self.rule]

    def before_run(self, runner):
        if self.save_best is not None:
            if runner.meta is None:
                warnings.warn('runner.meta is None. Creating a empty one.')
                runner.meta = dict()
            runner.meta.setdefault('hook_msgs', dict())

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training."""
        if not self.initial_epoch_flag:
            return
        if self.start is not None and runner.epoch >= self.start:
            self.after_train_epoch(runner)
        self.initial_epoch_flag = False

    def evaluation_flag(self, runner):
        """Judge whether to perform_evaluation after this epoch.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.start is None:
            if not self.every_n_epochs(runner, self.interval):
                # No evaluation during the interval epochs.
                return False
        elif (runner.epoch + 1) < self.start:
            # No evaluation if start is larger than the current epoch.
            return False
        else:
            # Evaluation only at epochs 3, 5, 7... if start==3 and interval==2
            if (runner.epoch + 1 - self.start) % self.interval:
                return False
        return True

    def after_train_epoch(self, runner):
        if not self.evaluation_flag(runner):
            return
        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            best_score = runner.meta['hook_msgs'].get(
                'best_score', self.init_value_map[self.rule])
            if self.compare_func(key_score, best_score):
                best_score = key_score
                runner.meta['hook_msgs']['best_score'] = best_score
                last_ckpt = runner.meta['hook_msgs']['last_ckpt']
                runner.meta['hook_msgs']['best_ckpt'] = last_ckpt
                mmcv.symlink(
                    last_ckpt,
                    osp.join(runner.work_dir,
                             f'best_{self.key_indicator}.pth'))
                self.logger.info(
                    f'Now best checkpoint is epoch_{runner.epoch + 1}.pth.'
                    f'Best {self.key_indicator} is {best_score:0.4f}')

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        if self.save_best is not None:
            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]
        else:
            return None


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Notes:
        If new arguments are added, tools/test.py may be effected.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        start (int, optional): Evaluation starting epoch. It enables evaluation
            before the training starts if ``start`` <= the resuming epoch.
            If None, whether to evaluate is merely decided by ``interval``.
            Default: None.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be save in best.json.
            Options are the evaluation metrics to the test dataset. e.g.,
            ``bbox_mAP``, ``segm_mAP`` for bbox detection and instance
            segmentation. ``AR@100`` for proposal recall. If ``save_best`` is
            ``auto``, the first key will be used. The interval of
            ``CheckpointHook`` should device EvalHook. Default: None.
        rule (str | None): Comparison rule for best score. If set to None,
            it will infer a reasonable rule. Default: 'None'.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 tmpdir=None,
                 gpu_collect=False,
                 save_best=None,
                 rule=None,
                 **eval_kwargs):
        super().__init__(
            dataloader,
            start=start,
            interval=interval,
            save_best=save_best,
            rule=rule,
            **eval_kwargs)
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

    def after_train_epoch(self, runner):
        if not self.evaluation_flag(runner):
            return

        from mmdet.apis import multi_gpu_test
        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            key_score = self.evaluate(runner, results)
            if self.save_best:
                best_score = runner.meta['hook_msgs'].get(
                    'best_score', self.init_value_map[self.rule])
                if self.compare_func(key_score, best_score):
                    best_score = key_score
                    runner.meta['hook_msgs']['best_score'] = best_score
                    last_ckpt = runner.meta['hook_msgs']['last_ckpt']
                    runner.meta['hook_msgs']['best_ckpt'] = last_ckpt
                    mmcv.symlink(
                        last_ckpt,
                        osp.join(runner.work_dir,
                                 f'best_{self.key_indicator}.pth'))
                    self.logger.info(
                        f'Now best checkpoint is {last_ckpt}.'
                        f'Best {self.key_indicator} is {best_score:0.4f}')
