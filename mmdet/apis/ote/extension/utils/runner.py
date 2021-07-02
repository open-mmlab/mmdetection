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

import time
import warnings

import mmcv
from mmcv.runner.utils import get_host_info
from mmcv.runner import RUNNERS, EpochBasedRunner, IterBasedRunner, IterLoader


@RUNNERS.register_module()
class EpochRunnerWithCancel(EpochBasedRunner):
    """
    Simple modification to EpochBasedRunner to allow cancelling the training during an epoch. The cancel training hook
    should set the runner.should_stop flag to True if stopping is required.

    # TODO: Implement cancelling of training via keyboard interrupt signal, instead of should_stop
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


@RUNNERS.register_module()
class IterBasedRunnerWithCancel(IterBasedRunner):
    """
    Simple modification to IterBasedRunner to allow cancelling the training. The cancel training hook
    should set the runner.should_stop flag to True if stopping is required.

    # TODO: Implement cancelling of training via keyboard interrupt signal, instead of should_stop
    """

    def main_loop(self, workflow, iter_loaders, **kwargs):
        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)
                    if self.should_stop:
                        return

    def run(self, data_loaders, workflow, max_iters=None, **kwargs):
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        self.should_stop = False
        self.main_loop(workflow, iter_loaders, **kwargs)
        self.should_stop = False

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')
