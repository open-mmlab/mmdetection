import argparse
import copy
import mmcv
import numpy as np
import os
import os.path as osp
import pycocotools.mask as maskUtils
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv import Config
from mmcv.parallel import collate, scatter
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.apis.inference import LoadImage
from mmdet.core import BitmapMasks
from mmdet.datasets import build_dataset
from mmdet.datasets.pipelines import Compose
from mmdet.integration.nncf import check_nncf_is_enabled, get_nncf_metadata
from mmdet.models import TwoStageDetector, build_detector
from mmdet.utils import ExtendedDictAction, collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--tensorboard-dir', help='the dir to save tensorboard logs')
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
        '--update_config', nargs='+', action=ExtendedDictAction, help='arguments in dict')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def determine_max_batch_size(cfg, distributed, dataset_len_per_gpu):
    def get_fake_input(cfg, orig_img_shape=(128, 128, 3), device='cuda'):
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        data = dict(img=np.zeros(orig_img_shape, dtype=np.uint8))
        data = test_pipeline(data)
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]
        return data

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()

    if 'pipeline' in cfg.data.train:
        img_shape = [t for t in cfg.data.train.pipeline if t['type'] == 'Resize'][0]['img_scale']
    else:
        img_shape = [t for t in cfg.data.train.dataset.pipeline if t['type'] == 'Resize'][0][
            'img_scale']

    channels = 3

    fake_input = get_fake_input(cfg, orig_img_shape=list(img_shape) + [channels])
    img_shape = fake_input['img_metas'][0][0]['pad_shape']

    width, height = img_shape[0], img_shape[1]

    percentage = 0.9

    min_bs = 2
    max_bs = min(512, int(dataset_len_per_gpu / percentage) + 1)
    step = 1

    batch_size = min_bs
    for bs in range(min_bs, max_bs, step):
        try:
            gt_boxes = [torch.tensor([[0., 0., width, height]]).cuda() for _ in range(bs)]
            gt_labels = [torch.tensor([0], dtype=torch.long).cuda() for _ in range(bs)]
            img_metas = [fake_input['img_metas'][0][0] for _ in range(bs)]

            gt_masks = None

            if isinstance(model, TwoStageDetector) and model.roi_head.with_mask:
                rles = maskUtils.frPyObjects(
                    [[0.0, 0.0, width, 0.0, width, height, 0.0, height]], height, width)
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

    resulting_batch_size = int(batch_size * percentage)

    del model
    torch.cuda.empty_cache()

    if distributed:
        rank, world_size = get_dist_info()

        resulting_batch_size = torch.tensor(resulting_batch_size).cuda()
        dist.all_reduce(resulting_batch_size, torch.distributed.ReduceOp.MIN)
        print('rank', rank, 'resulting_batch_size', resulting_batch_size)

        resulting_batch_size = int(resulting_batch_size.cpu())
    else:
        print('resulting_batch_size', resulting_batch_size)

    return resulting_batch_size


def init_dist_cpu(launcher, backend, **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        dist.init_process_group(backend=backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg_samples_per_gpu = cfg.data.samples_per_gpu
    if args.update_config is not None:
        cfg.merge_from_dict(args.update_config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
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

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if torch.cuda.is_available():
            init_dist(args.launcher, **cfg.dist_params)
        else:
            cfg.dist_params['backend'] = 'gloo'
            init_dist_cpu(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    if args.tensorboard_dir is not None:
        hooks = [hook for hook in cfg.log_config.hooks if hook.type == 'TensorboardLoggerHook']
        if hooks:
            hooks[0].log_dir = args.tensorboard_dir
        else:
            logger.warning('Failed to find TensorboardLoggerHook')

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
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    if cfg.get('nncf_config'):
        check_nncf_is_enabled()
        logger.info('NNCF config: {}'.format(cfg.nncf_config))
        meta.update(get_nncf_metadata())

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    datasets = [build_dataset(cfg.data.train)]

    dataset_len_per_gpu = sum(len(dataset) for dataset in datasets)
    if distributed:
        dataset_len_per_gpu = dataset_len_per_gpu // get_dist_info()[1]
    assert dataset_len_per_gpu > 0
    if cfg.data.samples_per_gpu == 'auto':
        if torch.cuda.is_available():
            logger.info('Auto-selection of samples per gpu (batch size).')
            cfg.data.samples_per_gpu = determine_max_batch_size(cfg, distributed, dataset_len_per_gpu)
            logger.info(f'Auto selected batch size: {cfg.data.samples_per_gpu} {dataset_len_per_gpu}')
            cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
        else:
            logger.warning('Auto-selection of batch size is not implemented for CPU.')
            logger.warning('Setting batch size to value taken from configuration file.')
            cfg.data.samples_per_gpu = cfg_samples_per_gpu
    if dataset_len_per_gpu < cfg.data.samples_per_gpu:
        cfg.data.samples_per_gpu = dataset_len_per_gpu
        logger.warning(f'Decreased samples_per_gpu to: {cfg.data.samples_per_gpu} '
                       f'because of dataset length: {dataset_len_per_gpu} '
                       f'and gpus number: {get_dist_info()[1]}')

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
        # also save nncf status in the checkpoint -- it is important,
        # since it is used in wrap_nncf_model for loading NNCF-compressed models
        if cfg.get('nncf_config'):
            nncf_metadata = get_nncf_metadata()
            cfg.checkpoint_config.meta.update(nncf_metadata)
    else:
        # cfg.checkpoint_config is None
        assert not cfg.get('nncf_config'), (
                "NNCF is enabled, but checkpoint_config is not set -- "
                "cannot store NNCF metainfo into checkpoints")

    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    try:
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)
    except KeyboardInterrupt:
        print("Early Stopping triggered")


if __name__ == '__main__':
    main()
