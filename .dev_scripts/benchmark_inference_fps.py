# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv
from mmcv import Config, DictAction
from mmcv.runner import init_dist
from tools.analysis_tools.benchmark import repeat_measure_inference_speed


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet benchmark a model of FPS')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint_root', help='Checkpoint file root path')
    parser.add_argument(
        '--round-num',
        type=int,
        default=1,
        help='round a number to a given precision in decimal digits')
    parser.add_argument(
        '--repeat-num',
        type=int,
        default=1,
        help='num of repeat measure inference speed')
    parser.add_argument(
        '--out', type=str, help='output path of gathered fps to be stored')
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


def format_dict_to_markdown(result_dict):
    out_strs = [
        '| model | fps | times_pre_image(ms) |\n',
        '| ----- | ---- | ----- | \n'
    ]

    for cfg_name, value in result_dict.items():
        name = cfg_name.replace('configs/', '')
        if value == 0:
            out_strs.append(f'| {name} | 0 | 0 | \n')
            continue

        fps = value['fps']
        ms_times_pre_image = value['ms_times_pre_image']
        if isinstance(fps, list):
            mean_fps = value['mean_fps']
            mean_times_pre_image = value['mean_times_pre_image']
            fps = f'{fps}[{mean_fps}]'
            ms_times_pre_image = \
                f'{ms_times_pre_image}[{mean_times_pre_image}]'

        out_strs.append(f'| {name} | {fps} | {ms_times_pre_image} | \n')

    print(''.join(out_strs), flush=True)


if __name__ == '__main__':
    args = parse_args()
    assert args.round_num >= 0
    assert args.repeat_num >= 1

    config = Config.fromfile(args.config)

    if args.launcher == 'none':
        raise NotImplementedError('Only supports distributed mode')
    else:
        init_dist(args.launcher)

    result_dict = {}
    for model_key in config:
        model_infos = config[model_key]
        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            record_metrics = model_info['metric']
            cfg_path = model_info['config'].strip()
            cfg = Config.fromfile(cfg_path)
            checkpoint = osp.join(args.checkpoint_root,
                                  model_info['checkpoint'].strip())
            try:
                fps = repeat_measure_inference_speed(cfg, checkpoint,
                                                     args.max_iter,
                                                     args.log_interval,
                                                     args.fuse_conv_bn,
                                                     args.repeat_num)
                if args.repeat_num > 1:
                    fps_list = [round(fps_, args.round_num) for fps_ in fps]
                    times_pre_image_list = [
                        round(1000 / fps_, args.round_num) for fps_ in fps
                    ]
                    mean_fps = round(
                        sum(fps_list) / len(fps_list), args.round_num)
                    mean_times_pre_image = round(
                        sum(times_pre_image_list) / len(times_pre_image_list),
                        args.round_num)
                    print(
                        f'{cfg_path} '
                        f'Overall fps: {fps_list}[{mean_fps}] img / s, '
                        f'times per image: '
                        f'{times_pre_image_list}[{mean_times_pre_image}] '
                        f'ms / img',
                        flush=True)
                    result_dict[cfg_path] = dict(
                        fps=fps_list,
                        mean_fps=mean_fps,
                        ms_times_pre_image=times_pre_image_list,
                        mean_times_pre_image=mean_times_pre_image)
                else:
                    print(
                        f'{cfg_path} fps : {fps:.{args.round_num}f} img / s, '
                        f'times per image: {1000 / fps:.{args.round_num}f} '
                        f'ms / img',
                        flush=True)
                    result_dict[cfg_path] = dict(
                        fps=round(fps, args.round_num),
                        ms_times_pre_image=round(1000 / fps, args.round_num))
            except Exception as e:
                print(f'{config} error: {repr(e)}')
                result_dict[cfg_path] = 0

    if args.out:
        mmcv.mkdir_or_exist(args.out)
        mmcv.dump(result_dict, osp.join(args.out, 'batch_inference_fps.json'))

    format_dict_to_markdown(result_dict)
