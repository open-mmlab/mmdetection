# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import subprocess
from collections import OrderedDict

import torch
from mmengine.runner import CheckpointLoader

convert_dict_fpn = {
    'backbone.fpn_lateral3': 'neck.lateral_convs.0.conv',
    'backbone.fpn_lateral4': 'neck.lateral_convs.1.conv',
    'backbone.fpn_lateral5': 'neck.lateral_convs.2.conv',
    'backbone.fpn_output3': 'neck.fpn_convs.0.conv',
    'backbone.fpn_output4': 'neck.fpn_convs.1.conv',
    'backbone.fpn_output5': 'neck.fpn_convs.2.conv',
    'backbone.top_block.p6': 'neck.fpn_convs.3.conv',
    'backbone.top_block.p7': 'neck.fpn_convs.4.conv',
}

convert_dict_rpn = {
    'proposal_generator.centernet_head.bbox_tower.0':
    'rpn_head.reg_convs.0.conv',
    'proposal_generator.centernet_head.bbox_tower.1':
    'rpn_head.reg_convs.0.gn',
    'proposal_generator.centernet_head.bbox_tower.3':
    'rpn_head.reg_convs.1.conv',
    'proposal_generator.centernet_head.bbox_tower.4':
    'rpn_head.reg_convs.1.gn',
    'proposal_generator.centernet_head.bbox_tower.6':
    'rpn_head.reg_convs.2.conv',
    'proposal_generator.centernet_head.bbox_tower.7':
    'rpn_head.reg_convs.2.gn',
    'proposal_generator.centernet_head.bbox_tower.9':
    'rpn_head.reg_convs.3.conv',
    'proposal_generator.centernet_head.bbox_tower.10':
    'rpn_head.reg_convs.3.gn',
    'proposal_generator.centernet_head.bbox_pred': 'rpn_head.conv_reg',
    'proposal_generator.centernet_head.scales.0.scale':
    'rpn_head.scales.0.scale',
    'proposal_generator.centernet_head.scales.1.scale':
    'rpn_head.scales.1.scale',
    'proposal_generator.centernet_head.scales.2.scale':
    'rpn_head.scales.2.scale',
    'proposal_generator.centernet_head.scales.3.scale':
    'rpn_head.scales.3.scale',
    'proposal_generator.centernet_head.scales.4.scale':
    'rpn_head.scales.4.scale',
    'proposal_generator.centernet_head.agn_hm': 'rpn_head.conv_cls',
}

convert_dict_roi = {
    'roi_heads.box_head.0.fc1': 'roi_head.bbox_head.0.shared_fcs.0',
    'roi_heads.box_head.0.fc2': 'roi_head.bbox_head.0.shared_fcs.1',
    'roi_heads.box_head.1.fc1': 'roi_head.bbox_head.1.shared_fcs.0',
    'roi_heads.box_head.1.fc2': 'roi_head.bbox_head.1.shared_fcs.1',
    'roi_heads.box_head.2.fc1': 'roi_head.bbox_head.2.shared_fcs.0',
    'roi_heads.box_head.2.fc2': 'roi_head.bbox_head.2.shared_fcs.1',
    'roi_heads.box_predictor.0.freq_weight':
    'roi_head.bbox_head.0.freq_weight',
    'roi_heads.box_predictor.0.cls_score.zs_weight':
    'roi_head.bbox_head.0.fc_cls.zs_weight',
    'roi_heads.box_predictor.0.cls_score.linear':
    'roi_head.bbox_head.0.fc_cls.linear',
    'roi_heads.box_predictor.0.bbox_pred.0': 'roi_head.bbox_head.0.fc_reg.0',
    'roi_heads.box_predictor.0.bbox_pred.2': 'roi_head.bbox_head.0.fc_reg.2',
    'roi_heads.box_predictor.1.freq_weight':
    'roi_head.bbox_head.1.freq_weight',
    'roi_heads.box_predictor.1.cls_score.zs_weight':
    'roi_head.bbox_head.1.fc_cls.zs_weight',
    'roi_heads.box_predictor.1.cls_score.linear':
    'roi_head.bbox_head.1.fc_cls.linear',
    'roi_heads.box_predictor.1.bbox_pred.0': 'roi_head.bbox_head.1.fc_reg.0',
    'roi_heads.box_predictor.1.bbox_pred.2': 'roi_head.bbox_head.1.fc_reg.2',
    'roi_heads.box_predictor.2.freq_weight':
    'roi_head.bbox_head.2.freq_weight',
    'roi_heads.box_predictor.2.cls_score.zs_weight':
    'roi_head.bbox_head.2.fc_cls.zs_weight',
    'roi_heads.box_predictor.2.cls_score.linear':
    'roi_head.bbox_head.2.fc_cls.linear',
    'roi_heads.box_predictor.2.bbox_pred.0': 'roi_head.bbox_head.2.fc_reg.0',
    'roi_heads.box_predictor.2.bbox_pred.2': 'roi_head.bbox_head.2.fc_reg.2',
    'roi_heads.mask_head.mask_fcn1': 'roi_head.mask_head.convs.0.conv',
    'roi_heads.mask_head.mask_fcn2': 'roi_head.mask_head.convs.1.conv',
    'roi_heads.mask_head.mask_fcn3': 'roi_head.mask_head.convs.2.conv',
    'roi_heads.mask_head.mask_fcn4': 'roi_head.mask_head.convs.3.conv',
    'roi_heads.mask_head.deconv': 'roi_head.mask_head.upsample',
    'roi_heads.mask_head.predictor': 'roi_head.mask_head.conv_logits',
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
        new_v = v
        if 'backbone.bottom_up' in k:
            new_k = k.replace('backbone.bottom_up', 'backbone')
            # for Transformer backbone
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
            # for resnet
            if 'base.' in k:
                new_k = new_k.replace('base.', '')

        elif 'backbone.fpn' in k or 'backbone.top_block' in k:
            old_k = k.replace('.weight', '')
            old_k = old_k.replace('.bias', '')
            new_k = k.replace(old_k, convert_dict_fpn[old_k])
        elif 'proposal_generator' in k:
            old_k = k.replace('.weight', '')
            old_k = old_k.replace('.bias', '')
            new_k = k.replace(old_k, convert_dict_rpn[old_k])
        elif 'roi_heads' in k:
            old_k = k.replace('.weight', '')
            old_k = old_k.replace('.bias', '')
            new_k = k.replace(old_k, convert_dict_roi[old_k])
        else:
            print('skip:', k)
            continue

        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained eva '
        'models to mmpretrain style.')
    parser.add_argument(
        '--src',
        default='Detic_LbaseI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth',
        help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument(
        '--dst',
        default='detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis.pth',
        help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    weight = {}
    new_state_dict = convert(state_dict)
    if 'backbone.fc.weight' in new_state_dict.keys():
        del [new_state_dict['backbone.fc.weight']]
    if 'backbone.fc.bias' in new_state_dict.keys():
        del [new_state_dict['backbone.fc.bias']]
    weight['state_dict'] = new_state_dict
    torch.save(weight, args.dst)

    sha = subprocess.check_output(['sha256sum', args.dst]).decode()
    final_file = args.dst.replace('.pth', '') + '-{}.pth'.format(sha[:8])
    subprocess.Popen(['mv', args.dst, final_file])
    print(f'Done!!, save to {final_file}')


if __name__ == '__main__':
    main()
