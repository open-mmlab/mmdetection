import time

from mmcv.runner import RUNNERS, EpochBasedRunner


@RUNNERS.register_module()
class EpochRunnerWithCancel(EpochBasedRunner):
    """
    Simple modification to EpochBasedRunner to allow cancelling the training during an epoch. The cancel training hook
    should set the runner.should_stop flag to True if stopping is required.

    # TODO: Implement cancelling of training via keyboard interrupt signal, instead of should_stop
    #   JIRA ticket:
    #   https://cosmonio.atlassian.net/browse/NI-662?atlOrigin=eyJpIjoiMWZlMjEzMTliNTA3NDAxOTg0MTRjZmM3ZmFjM2ZhOTQiLCJwIjoiaiJ9
    """

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.should_stop = False
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True)
            self.call_hook('after_train_iter')
            self._iter += 1
            if self.should_stop:
                self.should_stop = False
                break

        self.call_hook('after_train_epoch')
        self._epoch += 1
