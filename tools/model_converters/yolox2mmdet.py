import argparse
import subprocess
from collections import OrderedDict

import torch


def convert(in_file, out_file):
    """Convert keys in checkpoints."""
    in_state_dict = torch.load(in_file, map_location='cpu')
    out_state_dict = OrderedDict()

    for key, val in in_state_dict['model'].items():
        new_key = key

        if key[:17] == 'backbone.backbone':
            new_key = 'backbone' + key[17:]
        elif key[:8] == 'backbone':
            new_key = 'neck' + key[8:]

        if key[:4] == 'head':
            new_key = 'bbox_head' + key[4:]

        if key != new_key:
            print(f'{key} -> {new_key}')

        out_state_dict[new_key] = val
    torch.save(out_state_dict, out_file)

    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    if out_file.endswith('.pth'):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file
    final_file = out_file_name + f'-{sha[:8]}.pth'
    subprocess.Popen(['mv', out_file, final_file])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', help='input checkpoint file')
    parser.add_argument('out_file', help='output checkpoint file')
    args = parser.parse_args()
    convert(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
