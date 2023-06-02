# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert benchmark model json to script')
    parser.add_argument(
        'txt_path', type=str, help='txt path output by benchmark_filter')
    parser.add_argument(
        '--run', action='store_true', help='run script directly')
    parser.add_argument(
        '--out', type=str, help='path to save model benchmark script')

    args = parser.parse_args()
    return args


def determine_gpus(cfg_name):
    gpus = 8
    gpus_pre_node = 8

    if cfg_name.find('16x') >= 0:
        gpus = 16
    elif cfg_name.find('4xb4') >= 0:
        gpus = 4
        gpus_pre_node = 4
    elif 'lad' in cfg_name:
        gpus = 2
        gpus_pre_node = 2

    return gpus, gpus_pre_node


def main():
    args = parse_args()
    if args.out:
        out_suffix = args.out.split('.')[-1]
        assert args.out.endswith('.sh'), \
            f'Expected out file path suffix is .sh, but get .{out_suffix}'
    assert args.out or args.run, \
        ('Please specify at least one operation (save/run/ the '
         'script) with the argument "--out" or "--run"')

    root_name = './tools'
    train_script_name = osp.join(root_name, 'slurm_train.sh')

    commands = []
    partition_name = 'PARTITION=$1 '
    commands.append(partition_name)
    commands.append('\n')

    work_dir = 'WORK_DIR=$2 '
    commands.append(work_dir)
    commands.append('\n')

    cpus_pre_task = 'CPUS_PER_TASK=${3:-4} '
    commands.append(cpus_pre_task)
    commands.append('\n')
    commands.append('\n')

    with open(args.txt_path, 'r') as f:
        model_cfgs = f.readlines()
        for i, cfg in enumerate(model_cfgs):
            cfg = cfg.strip()
            if len(cfg) == 0:
                continue
            # print cfg name
            echo_info = f'echo \'{cfg}\' &'
            commands.append(echo_info)
            commands.append('\n')

            fname, _ = osp.splitext(osp.basename(cfg))
            out_fname = '$WORK_DIR/' + fname

            gpus, gpus_pre_node = determine_gpus(cfg)
            command_info = f'GPUS={gpus}  GPUS_PER_NODE={gpus_pre_node}  ' \
                           f'CPUS_PER_TASK=$CPUS_PRE_TASK {train_script_name} '
            command_info += '$PARTITION '
            command_info += f'{fname} '
            command_info += f'{cfg} '
            command_info += f'{out_fname} '

            command_info += '--cfg-options default_hooks.checkpoint.' \
                            'max_keep_ckpts=1 '
            command_info += '&'

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
