# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import time

import mmcv
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist

from mmdet.apis.test import collect_results_cpu
from mmdet.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Collect image metas')
    parser.add_argument('config', help='Config file path')
    parser.add_argument(
        'out',
        help='The output image metas file name. The save dir is in the '
        'same directory as `dataset.ann_file` path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def single_collect_metas(data_loader):
    metas = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        meta = data['img_metas']
        metas.extend(meta)
        batch_size = len(meta)
        for _ in range(batch_size):
            prog_bar.update()
    return metas


def multi_collect_metas(data_loader, tmpdir=None):
    metas = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        meta = data['img_metas']
        metas.extend(meta)
        if rank == 0:
            batch_size = len(meta)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect metas from all ranks
    metas = collect_results_cpu(metas, len(dataset), tmpdir)
    return metas


def main():
    args = parse_args()
    assert args.out.endswith('pkl'), 'The output file name must be pkl suffix'
    cfg = Config.fromfile(args.config)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    samples_per_gpu = 1
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    if not distributed:
        metas = single_collect_metas(data_loader)
    else:
        metas = multi_collect_metas(data_loader)

    root_path = cfg.data.test.ann_file.rsplit('/', 1)[0]
    save_path = os.path.join(root_path, args.out)
    mmcv.dump(metas, save_path)
    print(f'save image meta file: {save_path}')


if __name__ == '__main__':
    main()
