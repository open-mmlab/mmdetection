from mmcv.runner import Hook
import torch


class MemHook(Hook):

    def after_train_iter(self, runner):
        if (runner.iter + 1) % 1000 == 0:
            mem_a = torch.cuda.max_memory_allocated()
            mem_c = torch.cuda.max_memory_cached()
            mem_a_mb = int(mem_a / (1024 * 1024))
            mem_c_mb = int(mem_c / (1024 * 1024))
            print('max memory: {}/{}'.format(mem_a_mb, mem_c_mb))
