from __future__ import division

import argparse
import copy
from mmcv import Config
from mmcv.runner import obj_from_dict

from mmdet import datasets, __version__
from mmdet.datasets import ConcatDataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args


def get_train_dataset(cfg):
    if isinstance(cfg.data.train['ann_file'], list) or isinstance(cfg.data.train['ann_file'], tuple):
        ann_files = cfg.data.train['ann_file']
        train_datasets = []
        for ann_file in ann_files:
            data_info = copy.deepcopy(cfg.data.train)
            data_info['ann_file'] = ann_file
            train_dset = obj_from_dict(data_info, datasets)            
            train_datasets.append(train_dset)
        if len(train_datasets) > 1:
            train_dataset = ConcatDataset(train_datasets)            
        else:
            train_dataset = train_datasets[0]
    else:
        train_dataset = obj_from_dict(cfg.data.train, datasets)
    return train_dataset


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    cfg.gpus = args.gpus
    if cfg.checkpoint_config is not None:
        # save mmdet version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=cfg.text)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    train_dataset = get_train_dataset(cfg)
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
