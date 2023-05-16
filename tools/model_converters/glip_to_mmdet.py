# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch
from mmengine.runner import CheckpointLoader

convert_dict_fpn = {
    'module.backbone.fpn.fpn_inner2': 'neck.lateral_convs.0.conv',
    'module.backbone.fpn.fpn_inner3': 'neck.lateral_convs.1.conv',
    'module.backbone.fpn.fpn_inner4': 'neck.lateral_convs.2.conv',
    'module.backbone.fpn.fpn_layer2': 'neck.fpn_convs.0.conv',
    'module.backbone.fpn.fpn_layer3': 'neck.fpn_convs.1.conv',
    'module.backbone.fpn.fpn_layer4': 'neck.fpn_convs.2.conv',
    'module.backbone.fpn.top_blocks.p6': 'neck.fpn_convs.3.conv',
    'module.backbone.fpn.top_blocks.p7': 'neck.fpn_convs.4.conv',
}


def convert_eva(ckpt):
    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        if 'module.backbone.body' in k:
            new_k = k.replace('module.backbone.body', 'backbone')
        elif 'module.backbone.fpn' in k:
            old_k = k.replace('.weight', '')
            old_k = old_k.replace('.bias', '')
            new_k = k.replace(old_k, convert_dict_fpn[old_k])
        elif 'module.language_backbone' in k:
            new_k = k.replace('module.language_backbone', 'language_model.language_backbone')
            if 'pooler' in k:
                continue
        elif 'module.rpn' in k:
            if 'module.rpn.head.scales' in k:
                new_k = k.replace('module.rpn.head.scales', 'bbox_head.head.scales')
            else:
                new_k = k.replace('module.rpn', 'bbox_head')

            if 'anchor_generator' in k and 'resizer' in k:
                continue
        else:
            print('skip:', k)
            continue
        new_ckpt[new_k] = v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained eva '
                    'models to mmpretrain style.')
    parser.add_argument('--src', default='/home/PJLAB/huanghaian/yolo/GLIP/glip_a_tiny_o365.pth',
                        help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('--dst', default='../../glip_t.pth', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    weight = convert_eva(state_dict)
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()
