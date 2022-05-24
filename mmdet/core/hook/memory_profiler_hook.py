# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class MemoryProfilerHook(Hook):
    """Memory profiler hook recording memory information including virtual
    memory, swap memory, and the memory of the current process.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval=50):
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

    def after_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
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
