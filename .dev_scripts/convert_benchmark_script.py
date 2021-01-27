import argparse
import os
import os.path as osp

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert benchmark model json to script')
    parser.add_argument(
        'json_path', type=str, help='json path output by benchmark_filter')
    parser.add_argument('partition', type=str, help='slurm partition name')
    parser.add_argument(
        '--run', action='store_true', help='run script directly')
    parser.add_argument(
        '--out', type=str, help='path to save model benchmark script')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.out:
        out_suffix = args.out.split('.')[-1]
        assert args.out.endswith('.sh'), \
            f'Expected out file path suffix is .sh, but get .{out_suffix}'
    assert args.out or args.run, \
        ('Please specify at least one operation (save/run/ the '
         'script) with the argument "--out" or "--run"')

    json_data = mmcv.load(args.json_path)
    model_cfgs = json_data['models']

    partition = args.partition  # cluster name

    root_name = './tools'
    train_script_name = osp.join(root_name, 'slurm_train.sh')
    # stdout is no output
    stdout_cfg = '>/dev/null'

    commands = []
    for i, cfg in enumerate(model_cfgs):
        # print cfg name
        echo_info = 'echo \'' + cfg + '\' &'
        commands.append(echo_info)
        commands.append('\n')

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

        commands.append(command_info)

        if i < len(model_cfgs):
            commands.append('\n')
    command_str = ''.join(commands)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(command_str)
    if args.run:
        os.system(command_str)


if __name__ == '__main__':
    main()
