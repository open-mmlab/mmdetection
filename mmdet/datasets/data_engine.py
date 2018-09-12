from functools import partial
import torch
from .coco import CocoDataset
from .collate import collate
from .sampler import GroupSampler, DistributedGroupSampler


def build_data(cfg, args):
    dataset = CocoDataset(**cfg)

    if args.dist:
        sampler = DistributedGroupSampler(dataset, args.img_per_gpu,
                                     args.world_size, args.rank)
        batch_size = args.img_per_gpu
        num_workers = args.data_workers
    else:
        sampler = GroupSampler(dataset, args.img_per_gpu)
        batch_size = args.world_size * args.img_per_gpu
        num_workers = args.world_size * args.data_workers

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.img_per_gpu,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=args.img_per_gpu),
        pin_memory=False)

    return loader
