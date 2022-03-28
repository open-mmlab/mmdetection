# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook

try:
    import psutil
    from memory_profiler import memory_usage
except ImportError:
    psutil = None
    memory_usage = None


@HOOKS.register_module()
class MemoryProfilerHook(Hook):
    """Memory profiler.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval=50):
        if psutil is None and memory_usage is None:
            raise RuntimeError('Please install psutil and memory_profiler.')
        elif psutil is None:
            raise RuntimeError('Please install psutil.')
        elif memory_usage is None:
            raise RuntimeError('Please install memory_profiler.')

        self.interval = interval

    def after_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            virtual_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()
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
                'MB, '
                f'used_swap_memory: {round(swap_memory.used / factor)} MB, '
                f'swap_memory_utilization: {swap_memory.percent} %, '
                'current_process_memory: '
                f'{round(memory_usage()[0] / factor)} MB')
