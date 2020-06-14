import time

from mmcv.runner import Hook
from mmcv.utils import print_log
from torch.utils.data import DataLoader

from .update_stats import update_bn_stats


class PreciseBNHook(Hook):
    """Precise BN hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, num_iters, interval=1):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.num_iters = num_iters

    def after_train_epoch(self, runner):
        if self.every_n_epochs(runner, self.interval):
            print_log(
                f'Running Precise BN for {self.num_iters} iterations',
                logger=runner.logger)
            update_bn_stats(
                runner.model,
                self.dataloader,
                self.num_iters,
                logger=runner.logger)
            print_log('BN stats updated', logger=runner.logger)
            time.sleep(2.)
