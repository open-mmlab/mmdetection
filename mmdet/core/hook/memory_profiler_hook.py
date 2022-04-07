# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook

try:
    from psutil import swap_memory, virtual_memory
except ImportError:
    virtual_memory = None
    swap_memory = None

try:
    from memory_profiler import memory_usage
except ImportError:
    memory_usage = None


@HOOKS.register_module()
class MemoryProfilerHook(Hook):
    """Memory profiler hook recording memory information: virtual memory, swap
    memory and memory of current process.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval=50):
        if virtual_memory is None and memory_usage is None:
            raise RuntimeError('psutil and memory_profiler are not installed, '
                               'please install them by:'
                               'pip install psutil memory_profiler')
        elif virtual_memory is None:
            raise RuntimeError('psutil is not installed, please install it by:'
                               'pip install psutil')
        elif memory_usage is None:
            raise RuntimeError(
                'memory_profiler is not installed, please install it by:'
                'pip install memory_profiler')

        self.interval = interval

    def after_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            virtual_memory_ = virtual_memory()
            swap_memory_ = swap_memory()
            factor = 1024 * 1024
            runner.logger.info(
                'Memory information '
                'available_memory: '
                f'{round(virtual_memory_.available / factor)} MB, '
                'used_memory: '
                f'{round(virtual_memory_.used / factor)} MB, '
                f'memory_utilization: {virtual_memory_.percent} %, '
                'available_swap_memory: '
                f'{round((swap_memory_.total - swap_memory_.used) / factor)}'
                'MB, '
                f'used_swap_memory: {round(swap_memory_.used / factor)} MB, '
                f'swap_memory_utilization: {swap_memory_.percent} %, '
                'current_process_memory: '
                f'{round(memory_usage()[0] / factor)} MB')
