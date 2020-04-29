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
        # conv_cls for softmax output
        if out_channels != num_classes and out_channels % num_classes == 0:
            new_val = val.reshape(-1, num_classes, in_channels, *val.shape[2:])
            new_val = torch.cat((new_val[:, 1:], new_val[:, :1]), dim=1)
            new_val = new_val.reshape(val.size())
        # fc_cls
        elif out_channels == num_classes:
            new_val = torch.cat((val[1:], val[:1]), dim=0)
        # agnostic | retina_cls | rpn_cls
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


def convert(in_file,
            out_file,
            num_classes,
            upgrade_retina=False,
            is_ssd=False,
            reg_cls_agnostic=False):
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
        m = re.search(
            r'(conv_cls|retina_cls|rpn_cls|fc_cls|fcos_cls|'
            r'fovea_cls).(weight|bias)', new_key)
        if m is not None:
            print(f'reorder cls channels of {new_key}')
            new_val = reorder_cls_channel(val, num_classes)

        # regression
        m = re.search(r'(fc_reg|rpn_reg).(weight|bias)', new_key)
        if m is not None and not reg_cls_agnostic:
            print(f'truncate regression channels of {new_key}')
            new_val = truncate_reg_channel(val, num_classes)

        # mask head
        m = re.search(r'(conv_logits).(weight|bias)', new_key)
        if m is not None:
            print(f'truncate mask prediction channels of {new_key}')
            new_val = truncate_cls_channel(val, num_classes)

        m = re.search(r'(cls_convs|reg_convs).\d.(weight|bias)', key)
        # Legacy issues in RetinaNet since V1.x
        # Use ConvModule instead of nn.Conv2d in RetinaNet
        # cls_convs.0.weight -> cls_convs.0.conv.weight
        if m is not None and upgrade_retina:
            param = m.groups()[1]
            new_key = key.replace(param, 'conv.{}'.format(param))
            out_state_dict[new_key] = val
            print(f'rename the name of {key} to {new_key}')
            continue

        m = re.search(r'(cls_convs).\d.(weight|bias)', key)
        if m is not None and is_ssd:
            print(f'reorder cls channels of {new_key}')
            new_val = reorder_cls_channel(val, num_classes)

        out_state_dict[new_key] = new_val
    checkpoint['state_dict'] = out_state_dict
    torch.save(checkpoint, out_file)


def main():
    parser = argparse.ArgumentParser(description='Upgrade model version')
    parser.add_argument('in_file', help='input checkpoint file')
    parser.add_argument('out_file', help='output checkpoint file')
    parser.add_argument(
        '--num-classes',
        type=int,
        default=81,
        help='number of classes of the original model')
    parser.add_argument(
        '--upgrade-retina',
        action='store_true',
        help='whether to upgrade the retina head')
    parser.add_argument(
        '--ssd',
        action='store_true',
        help='whether to upgrade the SSD detctor')
    parser.add_argument(
        '--reg-cls-agnostic',
        action='store_true',
        help='whether the bbox regression is class agnostic '
        '(Cascade methods and SSD)')
    args = parser.parse_args()
    convert(args.in_file, args.out_file, args.num_classes, args.upgrade_retina,
            args.ssd, args.reg_cls_agnostic)


if __name__ == '__main__':
    main()
