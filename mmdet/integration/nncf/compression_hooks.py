from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.runner.dist_utils import master_only


@HOOKS.register_module()
class CompressionHook(Hook):
    def __init__(self, compression_ctrl=None):
        self.compression_ctrl = compression_ctrl

    def after_train_iter(self, runner):
        self.compression_ctrl.scheduler.step()

    def after_train_epoch(self, runner):
        self.compression_ctrl.scheduler.epoch_step()

    def before_run(self, runner):
        if runner.rank == 0:
            print_statistics(self.compression_ctrl.statistics(), runner.logger)


@HOOKS.register_module()
class CheckpointHookBeforeTraining(Hook):
    """Save checkpoints before training.

    Args:
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default.
    """

    def __init__(self,
                 save_optimizer=True,
                 out_dir=None,
                 **kwargs):
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs

    @master_only
    def before_run(self, runner):
        runner.logger.info(f'Saving checkpoint before training')
        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir, filename_tmpl='before_training.pth', save_optimizer=self.save_optimizer, **self.args)


def print_statistics(stats, logger):
    try:
        from texttable import Texttable
        texttable_imported = True
    except ImportError:
        texttable_imported = False

    for key, val in stats.items():
        if texttable_imported and isinstance(val, Texttable):
            logger.info(key)
            logger.info(val.draw())
        else:
            logger.info('{}: {}'.format(key, val))
