# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import time

import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import init_dist, load_checkpoint, wrap_fp16_model

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import replace_cfg_vals, update_data_root


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
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


def measure_inference_speed(cfg, checkpoint, max_iter, log_interval,
                            is_fuse_conv_bn):
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        # Because multiple processes will occupy additional CPU resources,
        # FPS statistics will be more unstable when workers_per_gpu is not 0.
        # It is reasonable to set workers_per_gpu to 0.
        workers_per_gpu=0,
        dist=True,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    if is_fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps


def repeat_measure_inference_speed(cfg,
                                   checkpoint,
                                   max_iter,
                                   log_interval,
                                   is_fuse_conv_bn,
                                   repeat_num=1):
    assert repeat_num >= 1

    fps_list = []

    for _ in range(repeat_num):
        #
        cp_cfg = copy.deepcopy(cfg)

        fps_list.append(
            measure_inference_speed(cp_cfg, checkpoint, max_iter, log_interval,
                                    is_fuse_conv_bn))

    if repeat_num > 1:
        fps_list_ = [round(fps, 1) for fps in fps_list]
        times_pre_image_list_ = [round(1000 / fps, 1) for fps in fps_list]
        mean_fps_ = sum(fps_list_) / len(fps_list_)
        mean_times_pre_image_ = sum(times_pre_image_list_) / len(
            times_pre_image_list_)
        print(
            f'Overall fps: {fps_list_}[{mean_fps_:.1f}] img / s, '
            f'times per image: '
            f'{times_pre_image_list_}[{mean_times_pre_image_:.1f}] ms / img',
            flush=True)
        return fps_list

    return fps_list[0]


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.launcher == 'none':
        raise NotImplementedError('Only supports distributed mode')
    else:
        init_dist(args.launcher, **cfg.dist_params)

    repeat_measure_inference_speed(cfg, args.checkpoint, args.max_iter,
                                   args.log_interval, args.fuse_conv_bn,
                                   args.repeat_num)


if __name__ == '__main__':
    main()
