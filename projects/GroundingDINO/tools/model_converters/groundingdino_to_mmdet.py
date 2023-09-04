# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import subprocess
from collections import OrderedDict

import torch
from mmengine.runner import CheckpointLoader


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
        new_v = v
        if 'module.backbone.0' in k:
            new_k = k.replace('module.backbone.0', 'backbone')
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
        elif 'module.bert' in k:
            new_k = k.replace('module.bert',
                              'language_model.language_backbone.body.model')
            # if 'pooler' in k:
            #     new_k = new_k.replace('pooler', 'pooler.dense')
        elif 'module.transformer.encoder' in k:
            new_k = k.replace('module.transformer.encoder', 'encoder')
            if 'norm1' in new_k:
                new_k = new_k.replace('norm1', 'norms.0')
            if 'norm2' in new_k:
                new_k = new_k.replace('norm2', 'norms.1')
            if 'norm3' in new_k:
                new_k = new_k.replace('norm3', 'norms.2')
            if 'linear1' in new_k:
                new_k = new_k.replace('linear1', 'ffn.layers.0.0')
            if 'linear2' in new_k:
                new_k = new_k.replace('linear2', 'ffn.layers.1')
        elif 'module.transformer.decoder' in k:
            new_k = k.replace('module.transformer.decoder', 'decoder')
            if 'norm1' in new_k:
                new_k = new_k.replace('norm1', 'norms.0')
            if 'norm2' in new_k:
                new_k = new_k.replace('norm2', 'norms.1')
            if 'norm3' in new_k:
                new_k = new_k.replace('norm3', 'norms.2')
            if 'linear1' in new_k:
                new_k = new_k.replace('linear1', 'ffn.layers.0.0')
            if 'linear2' in new_k:
                new_k = new_k.replace('linear2', 'ffn.layers.1')
            if 'self_attn' in new_k:
                new_k = new_k.replace('self_attn', 'self_attn.attn')
        else:
            print('skip:', k)
            continue

        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys to mmdet style.')
    parser.add_argument(
        'src',
        default='groundingdino_swint_ogc.pth.pth',
        help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument(
        'dst',
        default='groundingdino_swint_ogc.pth_mmdet.pth',
        help='save path')
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
