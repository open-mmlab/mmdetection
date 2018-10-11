from __future__ import division

import argparse
from mmcv import Config
from mmcv.runner import obj_from_dict

from mmdet import datasets
from mmdet.api import train_detector
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to add a validate phase')
    parser.add_argument(
        '--gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    cfg.validate = args.validate
    cfg.gpus = args.gpus
    cfg.seed = args.seed
    cfg.launcher = args.launcher
    cfg.local_rank = args.local_rank
    # build model
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    train_dataset = obj_from_dict(cfg.data.train, datasets)
    train_detector(model, train_dataset, cfg)


if __name__ == '__main__':
    main()
