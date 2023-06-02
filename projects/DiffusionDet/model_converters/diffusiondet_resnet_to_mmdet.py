# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import numpy as np
import torch
from mmengine.fileio import load


def convert(src, dst):
    if src.endswith('pth'):
        src_model = torch.load(src)
    else:
        src_model = load(src)

    dst_state_dict = OrderedDict()
    for k, v in src_model['model'].items():
        key_name_split = k.split('.')
        if 'backbone.fpn_lateral' in k:
            lateral_id = int(key_name_split[-2][-1])
            name = f'neck.lateral_convs.{lateral_id - 2}.' \
                   f'conv.{key_name_split[-1]}'
        elif 'backbone.fpn_output' in k:
            lateral_id = int(key_name_split[-2][-1])
            name = f'neck.fpn_convs.{lateral_id - 2}.conv.' \
                   f'{key_name_split[-1]}'
        elif 'backbone.bottom_up.stem.conv1.norm.' in k:
            name = f'backbone.bn1.{key_name_split[-1]}'
        elif 'backbone.bottom_up.stem.conv1.' in k:
            name = f'backbone.conv1.{key_name_split[-1]}'
        elif 'backbone.bottom_up.res' in k:
            # weight_type = key_name_split[-1]
            res_id = int(key_name_split[2][-1]) - 1
            # deal with short cut
            if 'shortcut' in key_name_split[4]:
                if 'shortcut' == key_name_split[-2]:
                    name = f'backbone.layer{res_id}.' \
                           f'{key_name_split[3]}.downsample.0.' \
                           f'{key_name_split[-1]}'
                elif 'shortcut' == key_name_split[-3]:
                    name = f'backbone.layer{res_id}.' \
                           f'{key_name_split[3]}.downsample.1.' \
                           f'{key_name_split[-1]}'
                else:
                    print(f'Unvalid key {k}')
            # deal with conv
            elif 'conv' in key_name_split[-2]:
                conv_id = int(key_name_split[-2][-1])
                name = f'backbone.layer{res_id}.{key_name_split[3]}' \
                       f'.conv{conv_id}.{key_name_split[-1]}'
            # deal with BN
            elif key_name_split[-2] == 'norm':
                conv_id = int(key_name_split[-3][-1])
                name = f'backbone.layer{res_id}.{key_name_split[3]}.' \
                       f'bn{conv_id}.{key_name_split[-1]}'
            else:
                print(f'{k} is invalid')

        elif key_name_split[0] == 'head':
            # d2: head.xxx -> mmdet: bbox_head.xxx
            name = f'bbox_{k}'
        else:
            # some base parameters such as beta will not convert
            print(f'{k} is not converted!!')
            continue

        if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
            raise ValueError(
                'Unsupported type found in checkpoint! {}: {}'.format(
                    k, type(v)))
        if not isinstance(v, torch.Tensor):
            dst_state_dict[name] = torch.from_numpy(v)
        else:
            dst_state_dict[name] = v
    mmdet_model = dict(state_dict=dst_state_dict, meta=dict())
    torch.save(mmdet_model, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
