# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import subprocess

import torch
from mmengine.logging import print_log


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    parser.add_argument(
        '--save-keys',
        nargs='+',
        type=str,
        default=['meta', 'state_dict'],
        help='keys to save in the published checkpoint')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, out_file, save_keys=['meta', 'state_dict']):
    checkpoint = torch.load(in_file, map_location='cpu')

    # only keep `meta` and `state_dict` for smaller file size
    ckpt_keys = list(checkpoint.keys())
    for k in ckpt_keys:
        if k not in save_keys:
            print_log(
                f'Key `{k}` will be removed because it is not in '
                f'save_keys. If you want to keep it, '
                f'please set --save-keys.',
                logger='current')
            checkpoint.pop(k, None)

    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    if torch.__version__ >= '1.6':
        torch.save(checkpoint, out_file, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    if out_file.endswith('.pth'):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file
    final_file = out_file_name + f'-{sha[:8]}.pth'
    subprocess.Popen(['mv', out_file, final_file])
    print_log(
        f'The published model is saved at {final_file}.', logger='current')


def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file, args.save_keys)


if __name__ == '__main__':
    main()
