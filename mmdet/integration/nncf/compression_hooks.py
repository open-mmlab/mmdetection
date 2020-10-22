from mmcv.runner.hooks.hook import Hook


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
