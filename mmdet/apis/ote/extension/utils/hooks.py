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

import os
from math import inf

from mmcv.runner import BaseRunner, EpochBasedRunner, IterBasedRunner
from mmcv.runner.hooks import HOOKS, Hook


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

    def before_run(self, runner):
        """ Remove .stop_training file before training begins if it exists

        :param runner:
        """
        work_dir = runner.work_dir
        stop_filepath = os.path.join(work_dir, '.stop_training')
        if os.path.exists(stop_filepath):
            os.remove(stop_filepath)

    @staticmethod
    def _check_for_stop_signal(runner: BaseRunner):
        work_dir = runner.work_dir
        stop_filepath = os.path.join(work_dir, '.stop_training')
        if os.path.exists(stop_filepath):
            if isinstance(runner, EpochBasedRunner):
                epoch = runner.epoch
                runner._max_epochs = epoch  # Force runner to stop by pretending it has reached it's max_epoch
            elif isinstance(runner, IterBasedRunner):
                iter = runner.iter
                runner._max_iters = iter
            runner.should_stop = True  # Set this flag to true to stop the current training epoch

    def after_train_iter(self, runner: BaseRunner):
        if not self.every_n_iters(runner, self.interval):
            return
        self._check_for_stop_signal(runner)


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    greater_keys = ['mAP', 'acc', 'top', 'AR@', 'auc', 'precision']
    less_keys = ['loss']

    def __init__(self, interval, metric='mAP', rule=None, patience=3, min_delta=0.0):
        super().__init__()
        self.patience = patience
        self.interval = interval
        self.min_delta = min_delta
        self._init_rule(rule, metric)

        self.min_delta *= 1 if self.rule == 'greater' else -1
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
            if key_indicator != 'auto':
                if key_indicator in self.greater_keys:
                    rule = 'greater'
                elif key_indicator in self.less_keys:
                    rule = 'less'
                elif any(key in key_indicator for key in self.greater_keys):
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
        self.by_epoch = False if runner.max_epochs is None else True

    def after_train_iter(self, runner):
        """Called after every training iter to evaluate the results."""
        if not self.by_epoch:
            self._do_check_stopping(runner)

    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""
        if self.by_epoch:
            self._do_check_stopping(runner)

    def _do_check_stopping(self, runner):
        if not self._should_check_stopping(runner):
            return

        if runner.rank == 0:
            key_score = runner.log_buffer.output[self.key_indicator]
            if self.compare_func(key_score - self.min_delta, self.best_score):
                self.best_score = key_score
                self.wait_count = 0
            else:
                self.wait_count += 1
                if self.wait_count >= self.patience:
                    stop_point = runner.epoch if self.by_epoch else runner.iter
                    print(f"Early Stopping at :{stop_point} with best {self.key_indicator}: {self.best_score}")
                    self._create_stop_file(runner)

    def _should_check_stopping(self, runner):
        check_time = self.every_n_epochs if self.by_epoch else self.every_n_iters
        if not check_time(runner, self.interval):
            # No evaluation during the interval.
            return False
        return True

    def _create_stop_file(self, runner):
        """ Create an empty stop file. CancelTrainingHook check this file under work_dir in order to stop training
        across multiple training processes

        :param runner: MMDetection runner
        """
        stop_filepath = os.path.join(runner.work_dir, '.stop_training')
        open(stop_filepath, 'a').close()
