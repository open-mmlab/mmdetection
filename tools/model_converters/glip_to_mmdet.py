# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import subprocess
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


def correct_unfold_reduction_order(x):
    out_channel, in_channel = x.shape
    x = x.reshape(out_channel, 4, in_channel // 4)
    x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
    return x


def correct_unfold_norm_order(x):
    in_channel = x.shape[0]
    x = x.reshape(4, in_channel // 4)
    x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
    return x


def convert(ckpt):
    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        if 'anchor_generator' in k or 'resizer' in k or 'cls_logits' in k:
            continue

        new_v = v
        if 'module.backbone.body' in k:
            new_k = k.replace('module.backbone.body', 'backbone')
            if 'patch_embed.proj' in new_k:
                new_k = new_k.replace('patch_embed.proj',
                                      'patch_embed.projection')
            elif 'pos_drop' in new_k:
                new_k = new_k.replace('pos_drop', 'drop_after_pos')

            if 'layers' in new_k:
                new_k = new_k.replace('layers', 'stages')
                if 'mlp.fc1' in new_k:
                    new_k = new_k.replace('mlp.fc1', 'ffn.layers.0.0')
                elif 'mlp.fc2' in new_k:
                    new_k = new_k.replace('mlp.fc2', 'ffn.layers.1')
                elif 'attn' in new_k:
                    new_k = new_k.replace('attn', 'attn.w_msa')

                if 'downsample' in k:
                    if 'reduction.' in k:
                        new_v = correct_unfold_reduction_order(v)
                    elif 'norm.' in k:
                        new_v = correct_unfold_norm_order(v)

        elif 'module.backbone.fpn' in k:
            old_k = k.replace('.weight', '')
            old_k = old_k.replace('.bias', '')
            new_k = k.replace(old_k, convert_dict_fpn[old_k])
        elif 'module.language_backbone' in k:
            new_k = k.replace('module.language_backbone',
                              'language_model.language_backbone')
            if 'pooler' in k:
                continue
        elif 'module.rpn' in k:
            if 'module.rpn.head.scales' in k:
                new_k = k.replace('module.rpn.head.scales',
                                  'bbox_head.head.scales')
            else:
                new_k = k.replace('module.rpn', 'bbox_head')

            if 'anchor_generator' in k and 'resizer' in k:
                continue
        else:
            print('skip:', k)
            continue

        if 'DyConv' in new_k:
            new_k = new_k.replace('DyConv', 'dyconvs')

        if 'AttnConv' in new_k:
            new_k = new_k.replace('AttnConv', 'attnconv')

        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys to mmdet style.')
    parser.add_argument(
        'src', default='glip_a_tiny_o365.pth', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument(
        '--dst', default='glip_tiny_a_mmdet.pth', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    weight = convert(state_dict)
    torch.save(weight, args.dst)

    sha = subprocess.check_output(['sha256sum', args.dst]).decode()
    final_file = args.dst.replace('.pth', '') + '-{}.pth'.format(sha[:8])
    subprocess.Popen(['mv', args.dst, final_file])
    print(f'Done!!, save to {final_file}')


if __name__ == '__main__':
    main()
