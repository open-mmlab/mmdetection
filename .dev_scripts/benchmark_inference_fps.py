import argparse
import os
import os.path as osp

import mmcv
from mmcv import Config, DictAction
from mmcv.runner import init_dist
from tools.analysis_tools.benchmark import measure_inferense_speed


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
        '--out', type=str, help='output path of gathered fps to be stored')
    parser.add_argument(
        '--max-iter', type=int, default=400, help='num of max iter')
    parser.add_argument(
        '--log-interval', type=int, default=40, help='interval of logging')
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


if __name__ == '__main__':
    args = parse_args()
    assert args.round_num >= 0

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
                fps = measure_inferense_speed(cfg, checkpoint, args.max_iter,
                                              args.log_interval,
                                              args.fuse_conv_bn)
                print(
                    f'{cfg_path} fps : {fps:.{args.round_num}f} img / s, '
                    f'times per image: {1000/fps:.{args.round_num}f} ms / img',
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
