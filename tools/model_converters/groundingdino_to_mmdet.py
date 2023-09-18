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
        #
        if 'module' not in k:
            # NOTE: swin-b has no module prefix and swin-t has module prefix
            k = 'module.' + k
        if 'module.bbox_embed' in k:
            # NOTE: bbox_embed name is swin-b is different from swin-t
            k = k.replace('module.bbox_embed',
                          'module.transformer.decoder.bbox_embed')

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
            # new_k = k.replace('module.bert', 'bert')

        elif 'module.feat_map' in k:
            new_k = k.replace('module.feat_map', 'text_feat_map')

        elif 'module.input_proj' in k:
            new_k = k.replace('module.input_proj', 'neck.convs')
            if 'neck.convs.3' in new_k:
                # extra convs for 4th scale
                new_k = new_k.replace('neck.convs.3', 'neck.extra_convs.0')
            if '0.weight' in new_k:
                # 0.weight -> conv.weight
                new_k = new_k.replace('0.weight', 'conv.weight')
            if '0.bias' in new_k:
                # 0.bias -> conv.bias
                new_k = new_k.replace('0.bias', 'conv.bias')
            if '1.weight' in new_k:
                # 1.weight -> gn.weight
                new_k = new_k.replace('1.weight', 'gn.weight')
            if '1.bias' in new_k:
                # 1.bias -> gn.bias
                new_k = new_k.replace('1.bias', 'gn.bias')

        elif 'module.transformer.level_embed' in k:
            # module.transformer.level_embed -> level_embed
            new_k = k.replace('module.transformer.level_embed', 'level_embed')

        elif 'module.transformer.encoder' in k:
            # if '.layers' in k:
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

            if 'text_layers' in new_k and 'self_attn' in new_k:
                new_k = new_k.replace('self_attn', 'self_attn.attn')

        elif 'module.transformer.enc_output' in k:
            if 'module.transformer.enc_output' in k and 'norm' not in k:
                new_k = k.replace('module.transformer.enc_output',
                                  'memory_trans_fc')
            if 'module.transformer.enc_output_norm' in k:
                new_k = k.replace('module.transformer.enc_output_norm',
                                  'memory_trans_norm')

        elif 'module.transformer.enc_out_bbox_embed.layers' in k:
            # ugly version
            if 'module.transformer.enc_out_bbox_embed.layers.0' in k:
                new_k = k.replace(
                    'module.transformer.enc_out_bbox_embed.layers.0',
                    'bbox_head.reg_branches.6.0')
            if 'module.transformer.enc_out_bbox_embed.layers.1' in k:
                new_k = k.replace(
                    'module.transformer.enc_out_bbox_embed.layers.1',
                    'bbox_head.reg_branches.6.2')
            if 'module.transformer.enc_out_bbox_embed.layers.2' in k:
                new_k = k.replace(
                    'module.transformer.enc_out_bbox_embed.layers.2',
                    'bbox_head.reg_branches.6.4')

        elif 'module.transformer.tgt_embed' in k:
            new_k = k.replace('module.transformer.tgt_embed',
                              'query_embedding')

        elif 'module.transformer.decoder' in k:
            new_k = k.replace('module.transformer.decoder', 'decoder')
            if 'norm1' in new_k:
                # norm1 in official GroundingDINO is the third norm in decoder
                new_k = new_k.replace('norm1', 'norms.2')
            if 'catext_norm' in new_k:
                # catext_norm in official GroundingDINO is the
                # second norm in decoder
                new_k = new_k.replace('catext_norm', 'norms.1')
            if 'norm2' in new_k:
                # norm2 in official GroundingDINO is the first norm in decoder
                new_k = new_k.replace('norm2', 'norms.0')
            if 'norm3' in new_k:
                new_k = new_k.replace('norm3', 'norms.3')
            if 'ca_text' in new_k:
                new_k = new_k.replace('ca_text', 'cross_attn_text')
                if 'in_proj_weight' in new_k:
                    new_k = new_k.replace('in_proj_weight',
                                          'attn.in_proj_weight')
                if 'in_proj_bias' in new_k:
                    new_k = new_k.replace('in_proj_bias', 'attn.in_proj_bias')
                if 'out_proj.weight' in new_k:
                    new_k = new_k.replace('out_proj.weight',
                                          'attn.out_proj.weight')
                if 'out_proj.bias' in new_k:
                    new_k = new_k.replace('out_proj.bias',
                                          'attn.out_proj.bias')
            if 'linear1' in new_k:
                new_k = new_k.replace('linear1', 'ffn.layers.0.0')
            if 'linear2' in new_k:
                new_k = new_k.replace('linear2', 'ffn.layers.1')
            if 'self_attn' in new_k:
                new_k = new_k.replace('self_attn', 'self_attn.attn')
            if 'bbox_embed' in new_k:
                reg_layer_id = int(new_k.split('.')[2])
                linear_id = int(new_k.split('.')[4])
                weight_or_bias = new_k.split('.')[-1]
                new_k = 'bbox_head.reg_branches.' + \
                    str(reg_layer_id)+'.'+str(2*linear_id)+'.'+weight_or_bias

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
