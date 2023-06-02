# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample


@HOOKS.register_module()
class MemoryProfilerHook(Hook):
    """Memory profiler hook recording memory information including virtual
    memory, swap memory, and the memory of the current process.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval: int = 50) -> None:
        try:
            from psutil import swap_memory, virtual_memory
            self._swap_memory = swap_memory
            self._virtual_memory = virtual_memory
        except ImportError:
            raise ImportError('psutil is not installed, please install it by: '
                              'pip install psutil')

        try:
            from memory_profiler import memory_usage
            self._memory_usage = memory_usage
        except ImportError:
            raise ImportError(
                'memory_profiler is not installed, please install it by: '
                'pip install memory_profiler')

        self.interval = interval

    def _record_memory_information(self, runner: Runner) -> None:
        """Regularly record memory information.

        Args:
            runner (:obj:`Runner`): The runner of the training or evaluation
                process.
        """
        # in Byte
        virtual_memory = self._virtual_memory()
        swap_memory = self._swap_memory()
        # in MB
        process_memory = self._memory_usage()[0]
        factor = 1024 * 1024
        runner.logger.info(
            'Memory information '
            'available_memory: '
            f'{round(virtual_memory.available / factor)} MB, '
            'used_memory: '
            f'{round(virtual_memory.used / factor)} MB, '
            f'memory_utilization: {virtual_memory.percent} %, '
            'available_swap_memory: '
            f'{round((swap_memory.total - swap_memory.used) / factor)}'
            ' MB, '
            f'used_swap_memory: {round(swap_memory.used / factor)} MB, '
            f'swap_memory_utilization: {swap_memory.percent} %, '
            'current_process_memory: '
            f'{round(process_memory)} MB')

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        """Regularly record memory information.

        Args:
            runner (:obj:`Runner`): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            self._record_memory_information(runner)

    def after_val_iter(
            self,
            runner: Runner,
            batch_idx: int,
            data_batch: Optional[dict] = None,
            outputs: Optional[Sequence[DetDataSample]] = None) -> None:
        """Regularly record memory information.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
            outputs (Sequence[:obj:`DetDataSample`], optional):
                Outputs from model. Defaults to None.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            self._record_memory_information(runner)

    def after_test_iter(
            self,
            runner: Runner,
            batch_idx: int,
            data_batch: Optional[dict] = None,
            outputs: Optional[Sequence[DetDataSample]] = None) -> None:
        """Regularly record memory information.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
            outputs (Sequence[:obj:`DetDataSample`], optional):
                Outputs from model. Defaults to None.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            self._record_memory_information(runner)
