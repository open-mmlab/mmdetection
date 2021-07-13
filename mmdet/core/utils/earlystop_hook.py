from math import inf

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    # TODO: ['acc', 'top', 'AR@', 'auc', 'precision'] not supported yet
    # TODO: Distributed training not supported yet
    greater_keys = ['mAP']
    less_keys = ['loss']

    def __init__(self, interval, metric='mAP', rule=None, patience=3, min_delta=0.0):
        super().__init__()
        self.patience = patience
        self.interval = interval
        self.min_delta = min_delta
        self._init_rule(rule, metric)

        self.min_delta *= 1 if self.rule == 'greater' else -1
        self.should_stop = False
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
        """perform evaluation and save ckpt."""
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
                    self.stopped = runner.iter
                    raise KeyboardInterrupt(
                        f"Early Stopping at iter: {runner.iter} with best {self.key_indicator}: {self.best_score} \n"
                        f'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {self.best_score}')

    def _should_check_stopping(self, runner):
        check_time = self.every_n_epochs if self.by_epoch else self.every_n_iters
        if not check_time(runner, self.interval):
            # No evaluation during the interval.
            return False
        return True
