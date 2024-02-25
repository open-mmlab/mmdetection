import argparse

import torch


def convert_weights(input_path):
    result = torch.load(input_path)['model']
    ul = {}
    for key in sorted(result.keys()):
        if 'detr.detr.backbone.0.' in key:
            ref_key = key.replace('detr.detr.backbone.0.', '')
            if 'stem.' in ref_key:
                ref_key = ref_key.replace('stem.', '')
                if 'norm.' in key:
                    ref_key = ref_key.replace('conv1.norm.', 'bn1.')
            else:
                split_res = ref_key.split('.')
                split_res[1] = 'layer' + str(int(split_res[1][-1]) - 1)
                ref_key = '.'.join([split for split in split_res])
                if 'conv1.norm.' in ref_key:
                    ref_key = ref_key.replace('conv1.norm.', 'bn1.')
                elif 'conv2.norm.' in ref_key:
                    ref_key = ref_key.replace('conv2.norm.', 'bn2.')
                elif 'conv3.norm.' in ref_key:
                    ref_key = ref_key.replace('conv3.norm.', 'bn3.')
                elif 'shortcut.norm.' in ref_key:
                    ref_key = ref_key.replace('shortcut.norm.',
                                              'downsample.1.')
                elif 'shortcut.weight' in key:
                    ref_key = ref_key.replace('shortcut.weight',
                                              'downsample.0.weight')
        elif 'detr.detr.class_embed.' in key:
            ref_key = key.replace('detr.detr.class_embed.',
                                  'bbox_head.cls_branches.')
            if 'body.' in ref_key:
                ref_key = ref_key.replace('body.', '')
        elif 'detr.detr.bbox_embed' in key:
            ref_key = key.replace('detr.detr.bbox_embed.',
                                  'bbox_head.reg_branches.')
        elif 'detr.detr.iou_head.' in key:
            ref_key = key.replace('detr.detr.iou_head.',
                                  'bbox_head.iou_branches.')
        elif 'detr.mask_head.' in key:
            ref_key = key.replace('detr.mask_head.', 'bbox_head.mask_head.')
            if 'jia_dcn.' in ref_key:
                ref_key = ref_key.replace('jia_dcn.', 'dcn.')
        elif 'detr.reid_embed_head.' in key:
            if 'detr.reid_embed_head.0.' in key:
                ref_key = key.replace('detr.reid_embed_head.0.',
                                      'bbox_head.reid_branch.')
                if 'self_attn.' in ref_key:
                    ref_key = ref_key.replace('self_attn.', 'self_attn.attn.')
                if 'linear1.' in ref_key:
                    ref_key = ref_key.replace('linear1.', 'ffn.layers.0.0.')
                elif 'linear2.' in ref_key:
                    ref_key = ref_key.replace('linear2.', 'ffn.layers.1.')
                elif 'norm2.' in ref_key:
                    ref_key = ref_key.replace('norm2.', 'norms.0.')
                elif 'norm1.' in ref_key:
                    ref_key = ref_key.replace('norm1.', 'norms.1.')
                elif 'norm3.' in ref_key:
                    ref_key = ref_key.replace('norm3.', 'norms.2.')
            elif 'detr.reid_embed_head.1.' in key:
                ref_key = key.replace('detr.reid_embed_head.1.',
                                      'bbox_head.reid_branch.mlp.')
            elif 'detr.reid_embed_head.0.ref_point_head.' in key:
                ref_key = key.replace('detr.reid_embed_head.0.ref_point_head.',
                                      'bbox_head.reid_branch.ref_point_head.')
        elif 'detr.detr.transformer.decoder.layers.' in key:
            ref_key = key.replace('detr.detr.transformer.decoder.layers.',
                                  'decoder.layers.')
            if 'linear1.' in ref_key:
                ref_key = ref_key.replace('linear1.', 'ffn.layers.0.0.')
            elif 'linear2.' in ref_key:
                ref_key = ref_key.replace('linear2.', 'ffn.layers.1.')
            elif 'norm2.' in ref_key:
                ref_key = ref_key.replace('norm2.', 'norms.0.')
            elif 'norm1.' in ref_key:
                ref_key = ref_key.replace('norm1.', 'norms.1.')
            elif 'norm3.' in ref_key:
                ref_key = ref_key.replace('norm3.', 'norms.2.')
            elif 'self_attn.' in ref_key:
                ref_key = ref_key.replace('self_attn.', 'self_attn.attn.')
        elif 'detr.detr.transformer.decoder.ref_point_head.' in key:
            ref_key = key.replace(
                'detr.detr.transformer.decoder.ref_point_head.',
                'decoder.ref_point_head.')
        elif 'detr.detr.transformer.level_embed' in key:
            ref_key = key.replace('detr.detr.transformer.level_embed',
                                  'level_embed')
        elif 'detr.detr.input_proj.' in key:
            ref_key = key.replace('detr.detr.input_proj.0.0.',
                                  'neck.convs.0.conv.')
            ref_key = ref_key.replace('detr.detr.input_proj.0.1.',
                                      'neck.convs.0.gn.')
            ref_key = ref_key.replace('detr.detr.input_proj.1.0.',
                                      'neck.convs.1.conv.')
            ref_key = ref_key.replace('detr.detr.input_proj.1.1.',
                                      'neck.convs.1.gn.')
            ref_key = ref_key.replace('detr.detr.input_proj.2.0.',
                                      'neck.convs.2.conv.')
            ref_key = ref_key.replace('detr.detr.input_proj.2.1.',
                                      'neck.convs.2.gn.')
            ref_key = ref_key.replace('detr.detr.input_proj.3.0.',
                                      'neck.extra_convs.0.conv.')
            ref_key = ref_key.replace('detr.detr.input_proj.3.1.',
                                      'neck.extra_convs.0.gn.')
        elif 'detr.detr.transformer.enc_output.' in key:
            ref_key = key.replace('detr.detr.transformer.enc_output.',
                                  'memory_trans_fc.')
        elif 'detr.detr.transformer.enc_output_norm.' in key:
            ref_key = key.replace('detr.detr.transformer.enc_output_norm.',
                                  'memory_trans_norm.')
        elif 'detr.detr.transformer.tgt_embed.weight' in key:
            ref_key = key.replace('detr.detr.transformer.tgt_embed.weight',
                                  'query_embedding.weight')
        elif 'detr.resizer.' in key:
            ref_key = key.replace('detr.resizer.fc.', 'resizer.0.')
            ref_key = ref_key.replace('detr.resizer.layer_norm.', 'resizer.1.')
        elif 'text_encoder.body.' in key:
            ref_key = key.replace('text_encoder.body.', 'text_encoder.')
        elif 'detr.detr.transformer.encoder.layers.' in key:
            ref_key = key.replace('detr.detr.transformer.encoder.layers.',
                                  'vl_encoder.vision_layers.')
            if 'linear1.' in ref_key:
                ref_key = ref_key.replace('linear1.', 'ffn.layers.0.0.')
            elif 'linear2.' in ref_key:
                ref_key = ref_key.replace('linear2.', 'ffn.layers.1.')
            elif 'norm1.' in ref_key:
                ref_key = ref_key.replace('norm1.', 'norms.0.')
            elif 'norm2.' in ref_key:
                ref_key = ref_key.replace('norm2.', 'norms.1.')
        elif 'detr.detr.transformer.encoder.vl_layers.' in key:
            ref_key = key.replace('detr.detr.transformer.encoder.vl_layers.',
                                  'vl_encoder.vlfuse_layers.')
        elif 'detr.controller.' in key:
            ref_key = key.replace('detr.controller.', 'bbox_head.controller.')
        else:
            continue

        ul[ref_key] = result[key]

    res = {'state_dict': ul}

    return res


def parse_args():
    parser = argparse.ArgumentParser(description='convert weight')
    parser.add_argument('--original-weight', type=str)
    parser.add_argument('--out-weight', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ori_weight_name = args.original_weight
    out_name = args.out_weight

    out = convert_weights(ori_weight_name)
    torch.save(out, out_name)


if __name__ == '__main__':
    main()
