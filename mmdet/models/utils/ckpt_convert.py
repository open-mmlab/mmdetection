# Copyright (c) OpenMMLab. All rights reserved.

# This script consists of several convert functions which
# can modify the weights of model in original repo to be
# pre-trained weights.

from collections import OrderedDict

import torch


def pvt_convert(ckpt):
    new_ckpt = OrderedDict()
    # Process the concat between q linear weights and kv linear weights
    use_abs_pos_embed = False
    use_conv_ffn = False
    for k in ckpt.keys():
        if k.startswith('pos_embed'):
            use_abs_pos_embed = True
        if k.find('dwconv') >= 0:
            use_conv_ffn = True
    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        if k.startswith('norm.'):
            continue
        if k.startswith('cls_token'):
            continue
        if k.startswith('pos_embed'):
            stage_i = int(k.replace('pos_embed', ''))
            new_k = k.replace(f'pos_embed{stage_i}',
                              f'layers.{stage_i - 1}.1.0.pos_embed')
            if stage_i == 4 and v.size(1) == 50:  # 1 (cls token) + 7 * 7
                new_v = v[:, 1:, :]  # remove cls token
            else:
                new_v = v
        elif k.startswith('patch_embed'):
            stage_i = int(k.split('.')[0].replace('patch_embed', ''))
            new_k = k.replace(f'patch_embed{stage_i}',
                              f'layers.{stage_i - 1}.0')
            new_v = v
            if 'proj.' in new_k:
                new_k = new_k.replace('proj.', 'projection.')
        elif k.startswith('block'):
            stage_i = int(k.split('.')[0].replace('block', ''))
            layer_i = int(k.split('.')[1])
            new_layer_i = layer_i + use_abs_pos_embed
            new_k = k.replace(f'block{stage_i}.{layer_i}',
                              f'layers.{stage_i - 1}.1.{new_layer_i}')
            new_v = v
            if 'attn.q.' in new_k:
                sub_item_k = k.replace('q.', 'kv.')
                new_k = new_k.replace('q.', 'attn.in_proj_')
                new_v = torch.cat([v, ckpt[sub_item_k]], dim=0)
            elif 'attn.kv.' in new_k:
                continue
            elif 'attn.proj.' in new_k:
                new_k = new_k.replace('proj.', 'attn.out_proj.')
            elif 'attn.sr.' in new_k:
                new_k = new_k.replace('sr.', 'sr.')
            elif 'mlp.' in new_k:
                string = f'{new_k}-'
                new_k = new_k.replace('mlp.', 'ffn.layers.')
                if 'fc1.weight' in new_k or 'fc2.weight' in new_k:
                    new_v = v.reshape((*v.shape, 1, 1))
                new_k = new_k.replace('fc1.', '0.')
                new_k = new_k.replace('dwconv.dwconv.', '1.')
                if use_conv_ffn:
                    new_k = new_k.replace('fc2.', '4.')
                else:
                    new_k = new_k.replace('fc2.', '3.')
                string += f'{new_k} {v.shape}-{new_v.shape}'
        elif k.startswith('norm'):
            stage_i = int(k[4])
            new_k = k.replace(f'norm{stage_i}', f'layers.{stage_i - 1}.2')
            new_v = v
        else:
            new_k = k
            new_v = v
        new_ckpt[new_k] = new_v

    return new_ckpt


def swin_converter(ckpt):

    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt['backbone.' + new_k] = new_v

    return new_ckpt
