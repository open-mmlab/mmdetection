# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook
from torch import distributed as dist


@HOOKS.register_module()
class SyncRandomSizeHook(Hook):
    """Change and synchronize the random image size across ranks, currently
    used in YOLOX.

    Args:
        ratio_range (tuple[int]): Random ratio range. It will be multiplied
            by 32, and then change the dataset output image size.
            Default: (14, 26).
        img_scale (tuple[int]): Size of input image. Default: (640, 640).
        interval (int): The interval of change image size. Default: 10.
        device (torch.device | str): device for returned tensors.
            Default: 'cuda'.
    """

    def __init__(self,
                 ratio_range=(14, 26),
                 img_scale=(640, 640),
                 interval=10,
                 device='cuda'):
        self.rank, world_size = get_dist_info()
        self.is_distributed = world_size > 1
        self.ratio_range = ratio_range
        self.img_scale = img_scale
        self.interval = interval
        self.device = device

    def after_train_iter(self, runner):
        """Change the dataset output image size."""
        if self.ratio_range is not None and (runner.iter +
                                             1) % self.interval == 0:
            # Due to DDP and DP get the device behavior inconsistent,
            # so we did not get the device from runner.model.
            tensor = torch.LongTensor(2).to(self.device)

            if self.rank == 0:
                size_factor = self.img_scale[1] * 1. / self.img_scale[0]
                size = random.randint(*self.ratio_range)
                size = (int(32 * size), 32 * int(size * size_factor))
                tensor[0] = size[0]
                tensor[1] = size[1]

            if self.is_distributed:
                dist.barrier()
                dist.broadcast(tensor, 0)

            runner.data_loader.dataset.update_dynamic_scale(
                (tensor[0].item(), tensor[1].item()))
