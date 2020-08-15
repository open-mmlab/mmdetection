import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        start (int, optional): Evaluation starting epoch. It enables evaluation
            before the training starts if ``start`` <= the resuming epoch.
            If None, whether to evaluate is merely decided by``interval``.
            Default: None.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self, dataloader, interval=1, start=None, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.start = start
        self.eval_kwargs = eval_kwargs

    def before_run(self, runner):
        """Evaluate the model at the start of training."""
        if self.start is not None and runner.epoch >= self.start:
            self.after_train_epoch(runner)

    def after_train_epoch(self, runner):
        if self.start is not None and (runner.epoch + 1) < self.start:
            return
        if not self.every_n_epochs(runner, self.interval):
            return
        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        start (int, optional): Evaluation starting epoch. It enables evaluation
            before the training starts if ``start`` <= the resuming epoch.
            If None, whether to evaluate is merely decided by``interval``.
            Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 start=None,
                 gpu_collect=False,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.start = start
        self.gpu_collect = gpu_collect
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if self.start is not None and (runner.epoch + 1) < self.start:
            return
        if not self.every_n_epochs(runner, self.interval):
            return
        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
