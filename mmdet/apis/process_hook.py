from mmcv.runner.hooks import HOOKS, Hook
import random
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def random_resize(random_size, data_loader, rank, is_distributed, input_size):
    tensor = torch.LongTensor(2).cuda()

    if rank == 0:
        size_factor = input_size[1] * 1. / input_size[0]
        size = random.randint(*random_size)
        size = (int(32 * size), 32 * int(size * size_factor))
        tensor[0] = size[0]
        tensor[1] = size[1]

    if is_distributed:
        dist.barrier()
        dist.broadcast(tensor, 0)

    data_loader.dataset.input_size = (tensor[0].item(), tensor[1].item())
    return data_loader.dataset.input_size


@HOOKS.register_module()
class ProcessHook(Hook):
    def __init__(self, random_size=(14, 26), input_size=(640, 640), no_aug_epochs=15):
        self.rank, world_size = get_dist_info()
        self.is_distributed = world_size > 1
        self.random_size = random_size
        self.input_size = input_size
        self.no_aug_epochs = no_aug_epochs

    def after_train_iter(self, runner):
        progress_in_iter = runner.iter
        train_loader = runner.data_loader
        # random resizing
        if self.random_size is not None and (progress_in_iter + 1) % 10 == 0:
            input_size = random_resize(self.random_size, train_loader, self.rank, self.is_distributed, self.input_size)
            # print('now input size:', input_size)

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        train_loader = runner.data_loader
        model = runner.model.module
        if epoch + 1 == runner.max_epochs - self.no_aug_epochs:
            print("--->No mosaic aug now!")
            train_loader.dataset.mosaic = False
            print("--->Add additional L1 loss now!")
            model.head.use_l1 = True
