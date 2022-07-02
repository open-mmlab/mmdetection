# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
import warnings
from functools import partial

import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader

from .samplers import (ClassAwareSampler, DistributedGroupSampler,
                       DistributedSampler, GroupSampler, InfiniteBatchSampler,
                       InfiniteGroupBatchSampler)

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def _concat_dataset(cfg, default_args=None):
    from .dataset_wrappers import ConcatDataset
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    separate_eval = cfg.get('separate_eval', True)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if 'separate_eval' in data_cfg:
            data_cfg.pop('separate_eval')
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
                                   MultiImageMixDataset, RepeatDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        dataset = MultiImageMixDataset(**cp_cfg)
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)  # 一般情况下这是正常流程

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     runner_type='EpochBasedRunner',
                     persistent_workers=False,
                     class_aware_sampler=None,
                     **kwargs):
    """构建 PyTorch 数据加载器。
    在分布式训练中，每个 GPU/进程都有一个数据加载器。
    在非分布式训练中，所有 GPU 共享一个数据加载器。(在mmdet中多卡训练一定是分布式训练)

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): 每个 GPU 的 bs.
        workers_per_gpu (int): 每个 GPU 使用多少个子进程来加载数据(这个需要根据bs大小不同进行调节,过大过小都会降低训练速度).
        num_gpus (int): GPU 数量。仅用于非分布式训练(默认为1).
        dist (bool): 是否分布式训练/测试.
        shuffle (bool): 是否在每个 epoch中 打乱数据加载顺序.
        seed (int, Optional): 使用的随机数种子.
        runner_type (str): runner的类型.
        persistent_workers (bool): 使用数据集一次后，数据加载器不会关闭工作进程.
            这会让`Dataset`的实例进程保持活跃状态.该参数仅在 PyTorch>=1.7.0 时有效.
        class_aware_sampler (dict): 训练时是否使用 `ClassAwareSampler`.
        kwargs: 用于初始化 DataLoader 的任何关键字参数

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()  # 单卡->0,1 多卡->卡id,卡总数

    if dist:
        # 当模型为 DistributedDataParallel 时,dataloader 的 batch_size 为每个 GPU 上的训练样本数(samples_per_gpu)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # 由于mmdet多卡仅支持DistributedDataParallel,所以这里默认单卡,即 num_gpus=1
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    if runner_type == 'IterBasedRunner':  # 一般用于分割任务
        # 这是一个batch采样器，每次可以产生一个batch索引,它可以在 `DataParallel` 和 `DistributedDataParallel` 中使用
        if shuffle:
            batch_sampler = InfiniteGroupBatchSampler(
                dataset, batch_size, world_size, rank, seed=seed)
        else:
            batch_sampler = InfiniteBatchSampler(
                dataset,
                batch_size,
                world_size,
                rank,
                seed=seed,
                shuffle=False)
        batch_size = 1
        sampler = None  # batch_sampler与sampler的本质一样,都是获取数据索引,具体使用哪一种取决于runner类型
    else:
        if class_aware_sampler is not None:
            # ClassAwareSampler 可用于分布式和非分布式训练.
            num_sample_class = class_aware_sampler.get('num_sample_class', 1)
            sampler = ClassAwareSampler(
                dataset,
                samples_per_gpu,
                world_size,
                rank,
                seed=seed,
                num_sample_class=num_sample_class)
        elif dist:
            # DistributedGroupSampler 会打乱数据以满足每个 GPU 上的图像在同一组中
            if shuffle:
                sampler = DistributedGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
            else:
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed)
        else:
            sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_sampler = None

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if (TORCH_VERSION != 'parrots'
            and digit_version(TORCH_VERSION) >= digit_version('1.7.0')):
        kwargs['persistent_workers'] = persistent_workers
    elif persistent_workers is True:
        warnings.warn('persistent_workers is invalid because your pytorch '
                      'version is lower than 1.7.0')

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=kwargs.pop('pin_memory', False),
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
