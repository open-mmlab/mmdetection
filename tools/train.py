import argparse
import copy
import os
import os.path as osp
import tempfile
import time

import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter
from mmcv.runner import init_dist, get_dist_info
import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.distributed as dist

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.apis.inference import LoadImage
from mmdet.datasets import build_dataset
from mmdet.datasets.pipelines import Compose
from mmdet.core import BitmapMasks
from mmdet.models import build_detector, TwoStageDetector
from mmdet.utils import collect_env, get_root_logger

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--update_config', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def determine_max_batch_size(cfg, distributed):
    def get_fake_input(cfg, orig_img_shape=(128, 128, 3), device='cuda'):
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        data = dict(img=np.zeros(orig_img_shape, dtype=np.uint8))
        data = test_pipeline(data)
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]
        return data

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    if 'pipeline' in cfg.data.train:
        img_shape = [t for t in cfg.data.train.pipeline if t['type'] == 'Resize'][0]['img_scale']
    else:
        img_shape = [t for t in cfg.data.train.dataset.pipeline if t['type'] == 'Resize'][0]['img_scale']

    channels = 3

    fake_input = get_fake_input(cfg, orig_img_shape=list(img_shape) + [channels])
    img_shape = fake_input['img_metas'][0][0]['pad_shape']

    width, height = img_shape[0], img_shape[1]

    min_bs = 2
    max_bs = 512
    step = 2

    batch_size = min_bs
    for bs in range(min_bs, max_bs, step):
        try:
            gt_boxes = [torch.tensor([[0., 0., width, height]]).cuda() for _ in range(bs)]
            gt_labels = [torch.tensor([0], dtype=torch.long).cuda() for _ in range(bs)]
            img_metas = [fake_input['img_metas'][0][0] for _ in range(bs)]

            gt_masks = None

            if isinstance(model, TwoStageDetector):
                if model.roi_head.with_mask:
                    rles = maskUtils.frPyObjects([[0.0, 0.0, width, 0.0, width, height, 0.0, height]], height, width)
                    rle = maskUtils.merge(rles)
                    mask = maskUtils.decode(rle)
                    gt_masks = [BitmapMasks([mask], height, width) for _ in range(bs)]

            if gt_masks is None:
                model(torch.rand(bs, channels, height, width).cuda(), img_metas=img_metas,
                      gt_bboxes=gt_boxes, gt_labels=gt_labels)
            else:
                model(torch.rand(bs, channels, height, width).cuda(), img_metas=img_metas,
                      gt_bboxes=gt_boxes, gt_labels=gt_labels, gt_masks=gt_masks)

            batch_size = bs
        except RuntimeError as e:
            if str(e).startswith('CUDA out of memory'):
                break

            raise e

    del model

    torch.cuda.empty_cache()
    resulting_batch_size = int(batch_size * 0.9)

    if distributed:
        rank, world_size = get_dist_info()
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
        dist.barrier()
        with open(os.path.join(tmpdir, str(rank)), 'w') as dst_file:
            dst_file.write(str(resulting_batch_size))

        print('before rank', rank, 'resulting batch size ', resulting_batch_size)
        dist.barrier()
        if rank == 0:
            min_batch_size = 1e6
            for i in range(world_size):
                with open(os.path.join(tmpdir, '0')) as src_file:
                    min_batch_size = min(min_batch_size, int(src_file.readlines()[0].strip()))

            with open(os.path.join(tmpdir, '0'), 'w') as dst_file:
                dst_file.write(str(min_batch_size))

        dist.barrier()
        with open(os.path.join(tmpdir, '0')) as src_file:
            resulting_batch_size = int(src_file.readlines()[0].strip())

        print('rank', rank, 'resulting batch size ', resulting_batch_size)

    return resulting_batch_size


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.update_config is not None:
        cfg.merge_from_dict(args.update_config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    if cfg.data.samples_per_gpu == 'auto':
        cfg.data.samples_per_gpu = determine_max_batch_size(cfg, distributed)
        cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
