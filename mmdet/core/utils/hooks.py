import torch
from mmcv.runner import Hook


class EmptyCacheHook(Hook):

    def before_epoch(self, runner):
        torch.cuda.empty_cache()

    def after_epoch(self, runner):
        torch.cuda.empty_cache()
