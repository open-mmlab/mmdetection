import os

from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner import EpochBasedRunner


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
    def _check_for_stop_signal(runner: EpochBasedRunner):
        work_dir = runner.work_dir
        stop_filepath = os.path.join(work_dir, '.stop_training')
        if os.path.exists(stop_filepath):
            epoch = runner.epoch
            runner._max_epochs = epoch  # Force runner to stop by pretending it has reached it's max_epoch
            runner.should_stop = True  # Set this flag to true to stop the current training epoch
            os.remove(stop_filepath)

    def after_train_iter(self, runner: EpochBasedRunner):
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
