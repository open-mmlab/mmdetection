# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

from mmengine import MMLogger
from mmengine.config import Config, DictAction
from mmengine.dist import init_dist
from mmengine.registry import init_default_scope
from mmengine.utils import mkdir_or_exist

from mmdet.utils.benchmark import (DataLoaderBenchmark, DatasetBenchmark,
                                   InferenceBenchmark)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--task',
        choices=['inference', 'dataloader', 'dataset'],
        default='dataloader',
        help='Which task do you want to go to benchmark')
    parser.add_argument(
        '--repeat-num',
        type=int,
        default=1,
        help='number of repeat times of measurement for averaging the results')
    parser.add_argument(
        '--max-iter', type=int, default=2000, help='num of max iter')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--num-warmup', type=int, default=5, help='Number of warmup')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--dataset-type',
        choices=['train', 'val', 'test'],
        default='test',
        help='Benchmark dataset type. only supports train, val and test')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing '
        'benchmark metrics')
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
    return args


def inference_benchmark(args, cfg, distributed, logger):
    benchmark = InferenceBenchmark(
        cfg,
        args.checkpoint,
        distributed,
        args.fuse_conv_bn,
        args.max_iter,
        args.log_interval,
        args.num_warmup,
        logger=logger)
    return benchmark


def dataloader_benchmark(args, cfg, distributed, logger):
    benchmark = DataLoaderBenchmark(
        cfg,
        distributed,
        args.dataset_type,
        args.max_iter,
        args.log_interval,
        args.num_warmup,
        logger=logger)
    return benchmark


def dataset_benchmark(args, cfg, distributed, logger):
    benchmark = DatasetBenchmark(
        cfg,
        args.dataset_type,
        args.max_iter,
        args.log_interval,
        args.num_warmup,
        logger=logger)
    return benchmark


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    distributed = False
    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.get('env_cfg', {}).get('dist_cfg', {}))
        distributed = True

    log_file = None
    if args.work_dir:
        log_file = os.path.join(args.work_dir, 'benchmark.log')
        mkdir_or_exist(args.work_dir)

    logger = MMLogger.get_instance(
        'mmdet', log_file=log_file, log_level='INFO')

    benchmark = eval(f'{args.task}_benchmark')(args, cfg, distributed, logger)
    benchmark.run(args.repeat_num)


if __name__ == '__main__':
    main()
