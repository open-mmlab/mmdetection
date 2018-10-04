from functools import partial

from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader

from .collate import collate
from .sampler import GroupSampler, DistributedGroupSampler


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus,
                     dist=True,
                     **kwargs):
    if dist:
        rank, world_size = get_dist_info()
        sampler = DistributedGroupSampler(dataset, imgs_per_gpu, world_size,
                                          rank)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, imgs_per_gpu)
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    if not kwargs.get('shuffle', True):
        sampler = None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        **kwargs)

    return data_loader
