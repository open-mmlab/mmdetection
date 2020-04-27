import argparse
import re
from collections import OrderedDict

import torch


def is_head(key):
    valid_head_list = [
        'bbox_head', 'mask_head', 'semantic_head', 'grid_head', 'mask_iou_head'
    ]

    return any(key.startswith(h) for h in valid_head_list)


def find_rpn_head(state_dict):
    for key in state_dict.keys():
        if key.find('rpn_head') != -1:
            return True
    return False


def reorder_cls_channel(val, num_classes=81):
    # bias
    if val.dim() == 1:
        new_val = torch.cat((val[1:], val[:1]), dim=0)
    # weight
    else:
        out_channels, in_channels = val.shape[:2]
        # conv_cls | retina_cls | rpn_cls
        if out_channels != num_classes and out_channels % num_classes == 0:
            new_val = val.reshape(num_classes, -1, in_channels, *val.shape[2:])
            new_val = torch.cat((new_val[1:], new_val[:1]), dim=0)
            new_val = new_val.reshape(val.size())
        # fc_cls
        elif out_channels == num_classes:
            new_val = torch.cat((val[1:], val[:1]), dim=0)
        # agnostic
        else:
            new_val = val

    return new_val


def truncate_cls_channel(val, num_classes=81):

    # bias
    if val.dim() == 1:
        if val.size(0) % num_classes == 0:
            new_val = val[:num_classes - 1]
        else:
            new_val = val
    # weight
    else:
        out_channels, in_channels = val.shape[:2]
        # conv_logits
        if out_channels % num_classes == 0:
            new_val = val.reshape(num_classes, in_channels, *val.shape[2:])[1:]
            new_val = new_val.reshape(-1, *val.shape[1:])
        # agnostic
        else:
            new_val = val

    return new_val


def truncate_reg_channel(val, num_classes=81):
    # bias
    if val.dim() == 1:
        # fc_reg|rpn_reg
        if val.size(0) % num_classes == 0:
            new_val = val.reshape(num_classes, -1)[:num_classes - 1]
            new_val = new_val.reshape(-1)
        # agnostic
        else:
            new_val = val
    # weight
    else:
        out_channels, in_channels = val.shape[:2]
        # fc_reg|rpn_reg
        if out_channels % num_classes == 0:
            new_val = val.reshape(num_classes, -1, in_channels,
                                  *val.shape[2:])[1:]
            new_val = new_val.reshape(-1, *val.shape[1:])
        # agnostic
        else:
            new_val = val

    return new_val


def convert(in_file, out_file, num_classes):
    """Convert keys in checkpoints.

    There can be some breaking changes during the development of mmdetection,
    and this tool is used for upgrading checkpoints trained with old versions
    to the latest one.
    """
    checkpoint = torch.load(in_file)
    in_state_dict = checkpoint.pop('state_dict')
    out_state_dict = OrderedDict()

    is_two_stage = find_rpn_head(in_state_dict)

    for key, val in in_state_dict.items():
        new_key = key
        new_val = val
        if is_two_stage and is_head(key):
            new_key = 'roi_head.{}'.format(key)

        # classification
        m = re.search(r'(conv_cls|retina_cls|rpn_cls|fc_cls).(weight|bias)',
                      new_key)
        if m is not None:
            print(f'reorder cls channels of {new_key}')
            new_val = reorder_cls_channel(val, num_classes)

        # regression
        m = re.search(r'(fc_reg|rpn_reg).(weight|bias)', new_key)
        if m is not None:
            print(f'truncate regression channels of {new_key}')
            new_val = truncate_reg_channel(val, num_classes)

        # mask head
        m = re.search(r'(conv_logits).(weight|bias)', new_key)
        if m is not None:
            print(f'truncate mask prediction channels of {new_key}')
            new_val = truncate_cls_channel(val, num_classes)

        out_state_dict[new_key] = new_val
    checkpoint['state_dict'] = out_state_dict
    torch.save(checkpoint, out_file)


def main():
    parser = argparse.ArgumentParser(description='Upgrade model version')
    parser.add_argument('in_file', help='input checkpoint file')
    parser.add_argument('out_file', help='output checkpoint file')
    parser.add_argument('--num-classes', type=int, default=81)
    args = parser.parse_args()
    convert(args.in_file, args.out_file, args.num_classes)


if __name__ == '__main__':
    main()
