from __future__ import division

import argparse
import logging
import random
from collections import OrderedDict

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import Runner, obj_from_dict

from mmdet import datasets, __version__
from mmdet.core import (init_dist, DistOptimizerHook, DistSamplerSeedHook,
                        MMDataParallel, MMDistributedDataParallel,
                        CocoDistEvalRecallHook, CocoDistEvalmAPHook)
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, RPN


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def get_logger(log_level):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    logger = logging.getLogger()
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    cfg.gpus = args.gpus
    # save mmdet version in checkpoint as meta data
    cfg.checkpoint_config.meta = dict(
        mmdet_version=__version__, config=cfg.text)

    logger = get_logger(cfg.log_level)

    # set random seed if specified
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    # init distributed environment if necessary
    if args.launcher == 'none':
        dist = False
        logger.info('Non-distributed training.')
    else:
        dist = True
        init_dist(args.launcher, **cfg.dist_params)
        if torch.distributed.get_rank() != 0:
            logger.setLevel('ERROR')
        logger.info('Distributed training.')

    # prepare data loaders
    train_dataset = obj_from_dict(cfg.data.train, datasets)
    data_loaders = [
        build_dataloader(train_dataset, cfg.data.imgs_per_gpu,
                         cfg.data.workers_per_gpu, cfg.gpus, dist)
    ]

    # build model
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if dist:
        model = MMDistributedDataParallel(model.cuda())
    else:
        model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)

    # register hooks
    optimizer_config = DistOptimizerHook(
        **cfg.optimizer_config) if dist else cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    if dist:
        runner.register_hook(DistSamplerSeedHook())
        # register eval hooks
        if args.validate:
            if isinstance(model.module, RPN):
                runner.register_hook(CocoDistEvalRecallHook(cfg.data.val))
            elif cfg.data.val.type == 'CocoDataset':
                runner.register_hook(CocoDistEvalmAPHook(cfg.data.val))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


if __name__ == '__main__':
    main()
