from __future__ import division
import argparse
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
sys.path.append('/mnt/lustre/pangjiangmiao/sensenet_folder/mmcv')

import torch
import torch.multiprocessing as mp
from mmcv import Config
from mmcv.torchpack import Runner
from mmdet.core import (batch_processor, init_dist, broadcast_params,
                        DistOptimizerStepperHook, DistSamplerSeedHook)
from mmdet.datasets.data_engine import build_data
from mmdet.models import Detector
from mmdet.nn.parallel import MMDataParallel


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet train val detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--validate', action='store_true', help='validate')
    parser.add_argument(
        '--dist', action='store_true', help='distributed training or not')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    args = parser.parse_args()

    return args


args = parse_args()


def main():
    # Enable distributed training or not
    if args.dist:
        print('Enable distributed training.')
        mp.set_start_method("spawn", force=True)
        init_dist(
            args.world_size,
            args.rank,
            **cfg.dist_params)
    else:
        print('Disabled distributed training.')

    # Fetch config information
    cfg = Config.fromfile(args.config)
    # TODO more flexible
    args.img_per_gpu = cfg.img_per_gpu
    args.data_workers = cfg.data_workers

    # prepare training loader
    train_loader = [build_data(cfg.train_dataset, args)]
    if args.validate:
        val_loader = build_data(cfg.val_dataset, args)
        train_loader.append(val_loader)

    # build model
    model = Detector(**cfg.model, **cfg.meta_params)
    if args.dist:
        model = model.cuda()
        broadcast_params(model)
    else:
        device_ids = args.rank % torch.cuda.device_count()
        model = MMDataParallel(model, device_ids=device_ids).cuda()

    # register hooks
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    optimizer_stepper = DistOptimizerStepperHook(
        **cfg.grad_clip_config) if args.dist else cfg.grad_clip_config
    runner.register_training_hooks(cfg.lr_policy, optimizer_stepper,
                                   cfg.checkpoint_config, cfg.log_config)
    if args.dist:
        runner.register_hook(DistSamplerSeedHook())
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(train_loader, cfg.workflow, cfg.max_epoch, args=args)


if __name__ == "__main__":
    main()
