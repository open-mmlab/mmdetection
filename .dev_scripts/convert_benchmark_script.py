import argparse
import os.path as osp

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert benchmark model json to script')
    parser.add_argument(
        'json_path', type=str, help='json path output by benchmark_filter')
    parser.add_argument('partition', type=str, help='Slurm partition name')
    parser.add_argument(
        '--out',
        default='regression_benchmark_configs.sh',
        type=str,
        help='path to save model benchmark script')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    json_data = mmcv.load(args.json_path)
    model_cfgs = json_data['models']

    partition = args.partition  # cluster name

    root_name = './tools'
    train_script_name = osp.join(root_name, 'slurm_train.sh')
    # stdout is no output
    stdout_cfg = '>/dev/null'

    with open(args.out, 'w') as f:
        for i, cfg in enumerate(model_cfgs):
            # print cfg name
            echo_info = 'echo \'' + cfg + '\' &'
            f.write(echo_info)
            f.write('\n')

            fname, _ = osp.splitext(osp.basename(cfg))
            out_fname = osp.join(root_name, fname)
            # default setting
            command_info = 'GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ' \
                           + train_script_name + ' '
            command_info += partition + ' '
            command_info += fname + ' '
            command_info += cfg + ' '
            command_info += out_fname + ' '
            command_info += stdout_cfg + ' &'

            f.write(command_info)

            if i < len(model_cfgs):
                f.write('\n')


if __name__ == '__main__':
    main()
