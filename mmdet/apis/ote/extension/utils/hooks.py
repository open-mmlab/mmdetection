# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import math
import os
from math import inf
from collections import defaultdict

from mmcv.runner.hooks import HOOKS, Hook, LoggerHook, LrUpdaterHook
from mmcv.runner import BaseRunner, EpochBasedRunner
from mmcv.runner.dist_utils import master_only
from mmcv.utils import print_log
from sc_sdk.logging import logger_factory


logger = logger_factory.get_logger("OTEDetectionTask.Hooks")


@HOOKS.register_module()
class CancelTrainingHook(Hook):
    def __init__(self, interval: int = 5):
        """
        Periodically check whether whether a stop signal is sent to the runner during model training.
        Every 'check_interval' iterations, the work_dir for the runner is checked to see if a file '.stop_training'
        is present. If it is, training is stopped.

        :param interval: Period for checking for stop signal, given in iterations.

        """
        self.interval = interval

    @staticmethod
    def _check_for_stop_signal(runner: BaseRunner):
        work_dir = runner.work_dir
        stop_filepath = os.path.join(work_dir, '.stop_training')
        if os.path.exists(stop_filepath):
            if isinstance(runner, EpochBasedRunner):
                epoch = runner.epoch
                runner._max_epochs = epoch  # Force runner to stop by pretending it has reached it's max_epoch
            runner.should_stop = True  # Set this flag to true to stop the current training epoch
            os.remove(stop_filepath)

    def after_train_iter(self, runner: BaseRunner):
        if not self.every_n_iters(runner, self.interval):
            return
        self._check_for_stop_signal(runner)


@HOOKS.register_module()
class FixedMomentumUpdaterHook(Hook):
    def __init__(self):
        """
        This hook does nothing, as the momentum is fixed by default. The hook is here to streamline switching between
        different LR schedules.
        """
        pass

    def before_run(self, runner):
        pass


@HOOKS.register_module()
class EnsureCorrectBestCheckpointHook(Hook):
    def __init__(self):
        """
        This hook makes sure that the 'best_mAP' checkpoint points properly to the best model, even if the best model is
        created in the last epoch.
        """
        pass

    def after_run(self, runner):
        runner.call_hook('after_train_epoch')


@HOOKS.register_module()
class OTELoggerHook(LoggerHook):

    class Curve:
        def __init__(self):
            self.x = []
            self.y = []

        def __repr__(self):
            points = []
            for x, y in zip(self.x, self.y):
                points.append(f'({x},{y})')
            return 'curve[' + ','.join(points) + ']'

    def __init__(self,
                 curves=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.curves = curves if curves is not None else defaultdict(self.Curve)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=False)
        if runner.max_epochs is not None:
            normalized_iter = runner.max_epochs / runner.max_iters * self.get_iter(runner)
        else:
            normalized_iter = self.get_iter(runner)
        for tag, value in tags.items():
            self.curves[tag].x.append(normalized_iter)
            self.curves[tag].y.append(value)


@HOOKS.register_module()
class OTEProgressHook(Hook):
    def __init__(self, time_monitor, verbose=False):
        super().__init__()
        self.time_monitor = time_monitor
        self.verbose = verbose
        self.print_threshold = 1

    def before_run(self, runner):
        total_epochs = runner.max_epochs if runner.max_epochs is not None else 1
        self.time_monitor.total_epochs = total_epochs
        self.time_monitor.train_steps = runner.max_iters // total_epochs
        self.time_monitor.steps_per_epoch = self.time_monitor.train_steps + self.time_monitor.val_steps
        self.time_monitor.total_steps = math.ceil(self.time_monitor.steps_per_epoch * total_epochs)
        self.time_monitor.current_step = 0
        self.time_monitor.current_epoch = 0

    def before_epoch(self, runner):
        self.time_monitor.on_epoch_begin(runner.epoch)

    def after_epoch(self, runner):
        self.time_monitor.on_epoch_end(runner.epoch)

    def before_iter(self, runner):
        self.time_monitor.on_train_batch_begin(1)

    def after_iter(self, runner):
        self.time_monitor.on_train_batch_end(1)
        if self.verbose:
            progress = self.progress
            if progress >= self.print_threshold:
                logger.warning(f'training progress {progress:.0f}%')
                self.print_threshold = (progress + 10) // 10 * 10

    def before_val_iter(self, runner):
        self.time_monitor.on_test_batch_begin(1)

    def after_val_iter(self, runner):
        self.time_monitor.on_test_batch_end(1)

    @property
    def progress(self):
        return self.time_monitor.get_progress()


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """
    Cancel training when a metric has stopped improving.

    Early Stopping hook monitors a metric quantity and if no improvement is seen for a ‘patience’
    number of epochs, the training is cancelled.

    :param interval: the number of intervals for checking early stop. The interval number should be
                     the same as the evaluation interval - the `interval` variable set in
                     `evaluation` config.
    :param metric: the metric name to be monitored
    :param rule: greater or less.  In `less` mode, training will stop when the metric has stopped
                 decreasing and in `greater` mode it will stop when the metric has stopped
                 increasing.
    :param patience: Number of epochs with no improvement after which the training will be reduced.
                     For example, if patience = 2, then we will ignore the first 2 epochs with no
                     improvement, and will only cancel the training after the 3rd epoch if the
                     metric still hasn’t improved then
    :param iteration_patience: Number of iterations must be trained after the last improvement
                               before training stops. The same as patience but the training
                               continues if the number of iteration is lower than iteration_patience
                               This variable makes sure a model is trained enough for some
                               iterations after the last improvement before stopping.
    :param min_delta: Minimal decay applied to lr. If the difference between new and old lr is
                      smaller than eps, the update is ignored
    """
    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    less_keys = ['loss']

    def __init__(self,
                 interval: int,
                 metric: str = 'bbox_mAP',
                 rule: str = None,
                 patience: int = 5,
                 iteration_patience: int = 500,
                 min_delta: float = 0.0):
        super().__init__()
        self.patience = patience
        self.iteration_patience = iteration_patience
        self.interval = interval
        self.min_delta = min_delta
        self._init_rule(rule, metric)

        self.min_delta *= 1 if self.rule == 'greater' else -1
        self.last_iter = 0
        self.wait_count = 0
        self.best_score = self.init_value_map[self.rule]

    def _init_rule(self, rule, key_indicator):
        """Initialize rule, key_indicator, comparison_func, and best score.

        Here is the rule to determine which rule is used for key indicator
        when the rule is not specific:
        1. If the key indicator is in ``self.greater_keys``, the rule will be
           specified as 'greater'.
        2. Or if the key indicator is in ``self.less_keys``, the rule will be
           specified as 'less'.
        3. Or if the key indicator is equal to the substring in any one item
           in ``self.greater_keys``, the rule will be specified as 'greater'.
        4. Or if the key indicator is equal to the substring in any one item
           in ``self.less_keys``, the rule will be specified as 'less'.

        Args:
            rule (str | None): Comparison rule for best score.
            key_indicator (str | None): Key indicator to determine the
                comparison rule.
        """
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f'rule must be greater, less or None, '
                           f'but got {rule}.')

        if rule is None:
            if key_indicator in self.greater_keys or any(
                    key in key_indicator for key in self.greater_keys):
                rule = 'greater'
            elif key_indicator in self.less_keys or any(
                    key in key_indicator for key in self.less_keys):
                rule = 'less'
            else:
                raise ValueError(f'Cannot infer the rule for key '
                                 f'{key_indicator}, thus a specific rule '
                                 f'must be specified.')
        self.rule = rule
        self.key_indicator = key_indicator
        self.compare_func = self.rule_map[self.rule]

    def before_run(self, runner):
        self.by_epoch = False if runner.max_epochs is None else True
        for hook in runner.hooks:
            if isinstance(hook, LrUpdaterHook):
                self.warmup_iters = hook.warmup_iters
                break

    def after_train_iter(self, runner):
        """Called after every training iter to evaluate the results."""
        if not self.by_epoch:
            self._do_check_stopping(runner)

    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""
        if self.by_epoch:
            self._do_check_stopping(runner)

    def _do_check_stopping(self, runner):
        if not self._should_check_stopping(
                runner) or self.warmup_iters > runner.iter:
            return

        if runner.rank == 0:
            if self.key_indicator not in runner.log_buffer.output:
                raise KeyError(
                    f'metric {self.key_indicator} does not exist in buffer. Please check '
                    f'{self.key_indicator} is cached in evaluation output buffer'
                )

            key_score = runner.log_buffer.output[self.key_indicator]
            if self.compare_func(key_score - self.min_delta, self.best_score):
                self.best_score = key_score
                self.wait_count = 0
                self.last_iter = runner.iter
            else:
                self.wait_count += 1
                if self.wait_count >= self.patience:
                    if runner.iter - self.last_iter < self.iteration_patience:
                        print_log(
                            f"\nSkip early stopping. Accumulated iteration "
                            f"{runner.iter - self.last_iter} from the last "
                            f"improvement must be larger than {self.iteration_patience} to trigger "
                            f"Early Stopping.",
                            logger=runner.logger)
                        return
                    stop_point = runner.epoch if self.by_epoch else runner.iter
                    print_log(
                        f"\nEarly Stopping at :{stop_point} with "
                        f"best {self.key_indicator}: {self.best_score}",
                        logger=runner.logger)
                    runner.should_stop = True

    def _should_check_stopping(self, runner):
        check_time = self.every_n_epochs if self.by_epoch else self.every_n_iters
        if not check_time(runner, self.interval):
            # No evaluation during the interval.
            return False
        return True


@HOOKS.register_module()
class ReduceLROnPlateauLrUpdaterHook(LrUpdaterHook):
    """
    Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
    This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’
    number of epochs, the learning rate is reduced.

    :param min_lr: minimum learning rate. The lower bound of the desired learning rate.
    :param interval: the number of intervals for checking the hook. The interval number should be
                     the same as the evaluation interval - the `interval` variable set in
                     `evaluation` config.
    :param metric: the metric name to be monitored
    :param rule: greater or less.  In `less` mode, learning rate will be dropped if the metric has
                 stopped decreasing and in `greater` mode it will be dropped when the metric has
                 stopped increasing.
    :param patience: Number of epochs with no improvement after which learning rate will be reduced.
                     For example, if patience = 2, then we will ignore the first 2 epochs with no
                     improvement, and will only drop LR after the 3rd epoch if the metric still
                     hasn’t improved then
    :param iteration_patience: Number of iterations must be trained after the last improvement
                               before LR drops. The same as patience but the LR remains the same if
                               the number of iteration is lower than iteration_patience. This
                               variable makes sure a model is trained enough for some iterations
                               after the last improvement before dropping the LR.
    :param factor: Factor to be multiply with the learning rate.
                   For example, new_lr = current_lr * factor
    """
    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    less_keys = ['loss']

    def __init__(self,
                 min_lr,
                 interval,
                 metric='bbox_mAP',
                 rule=None,
                 factor=0.1,
                 patience=3,
                 iteration_patience=300,
                 **kwargs):
        super().__init__(**kwargs)
        self.interval = interval
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.iteration_patience = iteration_patience
        self.metric = metric
        self.bad_count = 0
        self.last_iter = 0
        self.current_lr = None
        self._init_rule(rule, metric)
        self.best_score = self.init_value_map[self.rule]

    def _init_rule(self, rule, key_indicator):
        """Initialize rule, key_indicator, comparison_func, and best score.

        Here is the rule to determine which rule is used for key indicator
        when the rule is not specific:
        1. If the key indicator is in ``self.greater_keys``, the rule will be
           specified as 'greater'.
        2. Or if the key indicator is in ``self.less_keys``, the rule will be
           specified as 'less'.
        3. Or if the key indicator is equal to the substring in any one item
           in ``self.greater_keys``, the rule will be specified as 'greater'.
        4. Or if the key indicator is equal to the substring in any one item
           in ``self.less_keys``, the rule will be specified as 'less'.

        Args:
            rule (str | None): Comparison rule for best score.
            key_indicator (str | None): Key indicator to determine the
                comparison rule.
        """
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f'rule must be greater, less or None, '
                           f'but got {rule}.')

        if rule is None:
            if key_indicator in self.greater_keys or any(
                    key in key_indicator for key in self.greater_keys):
                rule = 'greater'
            elif key_indicator in self.less_keys or any(
                    key in key_indicator for key in self.less_keys):
                rule = 'less'
            else:
                raise ValueError(f'Cannot infer the rule for key '
                                 f'{key_indicator}, thus a specific rule '
                                 f'must be specified.')
        self.rule = rule
        self.key_indicator = key_indicator
        self.compare_func = self.rule_map[self.rule]

    def _should_check_stopping(self, runner):
        check_time = self.every_n_epochs if self.by_epoch else self.every_n_iters
        if not check_time(runner, self.interval):
            # No evaluation during the interval.
            return False
        return True

    def get_lr(self, runner, base_lr):
        if not self._should_check_stopping(
                runner) or self.warmup_iters > runner.iter:
            return base_lr

        if self.current_lr is None:
            self.current_lr = base_lr

        if hasattr(runner, self.metric):
            score = getattr(runner, self.metric, 0.0)
        else:
            return self.current_lr

        print_log(
            f"\nBest Score: {self.best_score}, Current Score: {score}, Patience: {self.patience} "
            f"Count: {self.bad_count}",
            logger=runner.logger)
        if self.compare_func(score, self.best_score):
            self.best_score = score
            self.bad_count = 0
            self.last_iter = runner.iter
        else:
            self.bad_count += 1

        if self.bad_count >= self.patience:
            if runner.iter - self.last_iter < self.iteration_patience:
                print_log(
                    f"\nSkip LR dropping. Accumulated iteration "
                    f"{runner.iter - self.last_iter} from the last "
                    f"improvement must be larger than {self.iteration_patience} to trigger "
                    f"LR dropping.",
                    logger=runner.logger)
                return self.current_lr
            self.last_iter = runner.iter
            self.bad_count = 0
            print_log(
                f"\nDrop LR from: {self.current_lr}, to: "
                f"{max(self.current_lr * self.factor, self.min_lr)}",
                logger=runner.logger)
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
        return self.current_lr
