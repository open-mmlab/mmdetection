# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch
from mmengine.fileio import load


def convert_bn(blobs, state_dict, caffe_name, torch_name, converted_names):
    # detectron replace bn with affine channel layer
    state_dict[torch_name + '.bias'] = torch.from_numpy(blobs[caffe_name +
                                                              '_b'])
    state_dict[torch_name + '.weight'] = torch.from_numpy(blobs[caffe_name +
                                                                '_s'])
    bn_size = state_dict[torch_name + '.weight'].size()
    state_dict[torch_name + '.running_mean'] = torch.zeros(bn_size)
    state_dict[torch_name + '.running_var'] = torch.ones(bn_size)
    converted_names.add(caffe_name + '_b')
    converted_names.add(caffe_name + '_s')


def convert_conv_fc(blobs, state_dict, caffe_name, torch_name,
                    converted_names):
    state_dict[torch_name + '.weight'] = torch.from_numpy(blobs[caffe_name +
                                                                '_w'])
    converted_names.add(caffe_name + '_w')
    if caffe_name + '_b' in blobs:
        state_dict[torch_name + '.bias'] = torch.from_numpy(blobs[caffe_name +
                                                                  '_b'])
        converted_names.add(caffe_name + '_b')


def convert_patch_embed(model_key, model_weight, state_dict, converted_names):
    new_key = model_key.replace('proj', 'projection')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_backbone(model_key, model_weight, state_dict, converted_names):
    if 'patch_embed' in model_key:
        new_key = model_key.replace('proj', 'projection')
    else:
        new_key = model_key.replace('layers', 'stages')
        new_key = new_key.replace('attn', 'attn.w_msa')
        if 'mlp' in new_key:
            new_key = new_key.replace('mlp', 'ffn.layers')
            new_key = new_key.replace('fc1', '0.0')
            new_key = new_key.replace('fc2', '1')

    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_pixel_decoder(model_key, model_weight, state_dict,
                          converted_names):
    new_key = model_key.replace('sem_seg_head', 'panoptic_head')
    if 'input_proj' in new_key:
        spilt_key = new_key.split('.')
        if spilt_key[-2] == '1':
            module_name = 'gn'
        elif spilt_key[-2] == '0':
            module_name = 'conv'
        new_key = f'{spilt_key[0]}.{spilt_key[1]}.input_convs.{spilt_key[3]}.{module_name}.{spilt_key[5]}'

    new_key = new_key.replace('transformer.encoder', 'encoder')
    new_key = new_key.replace('linear1', 'ffn.layers.0.0')
    new_key = new_key.replace('linear2', 'ffn.layers.1')
    new_key = new_key.replace('norm1', 'norms.0')
    new_key = new_key.replace('norm2', 'norms.1')
    new_key = new_key.replace('transformer.level_embed',
                              'level_encoding.weight')
    new_key = new_key.replace('mask_features', 'mask_feature')
    new_key = new_key.replace('adapter_1.weight',
                              'lateral_convs.0.conv.weight')
    new_key = new_key.replace('adapter_1.norm', 'lateral_convs.0.gn')
    new_key = new_key.replace('layer_1.weight', 'output_convs.0.conv.weight')
    new_key = new_key.replace('layer_1.norm', 'output_convs.0.gn')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_class_transformer(model_key, model_weight, state_dict,
                              converted_names):
    new_key = model_key.replace('sem_seg_head.predictor', 'panoptic_head')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_predictor(model_key, model_weight, state_dict, converted_names):
    new_key = model_key.replace('sem_seg_head.predictor', 'panoptic_head')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_taskmlp(model_key, model_weight, state_dict, converted_names):
    new_key = model_key.replace('task_mlp', 'panoptic_head.task_mlp')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_text_encoder(model_key, model_weight, state_dict, converted_names):
    new_key = model_key.replace('text_encoder', 'panoptic_head.text_encoder')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_prompt_ctx(model_key, model_weight, state_dict, converted_names):
    new_key = model_key.replace('prompt_ctx', 'panoptic_head.prompt_ctx')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_text_projector(model_key, model_weight, state_dict,
                           converted_names):
    new_key = model_key.replace('text_projector',
                                'panoptic_head.text_projector')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_transformer_decoder(model_key, model_weight, state_dict,
                                converted_names):
    if 'transformer_self_attention_layers' in model_key:
        new_key = model_key.replace(
            'sem_seg_head.predictor.transformer_self_attention_layers',
            'panoptic_head.transformer_decoder.layers')
        new_key = new_key.replace('self_attn', 'self_attn.attn')
        new_key = new_key.replace(
            'norm', 'norms.1')  # mmdet中Decoder先放进来的是交叉注意力，与原始OneFormer相反
    elif 'transformer_cross_attention_layers' in model_key:
        new_key = model_key.replace(
            'sem_seg_head.predictor.transformer_cross_attention_layers',
            'panoptic_head.transformer_decoder.layers')
        new_key = new_key.replace('multihead_attn', 'cross_attn.attn')
        new_key = new_key.replace('norm', 'norms.0')
    elif 'transformer_ffn_layers' in model_key:
        new_key = model_key.replace(
            'sem_seg_head.predictor.transformer_ffn_layers',
            'panoptic_head.transformer_decoder.layers')
        new_key = new_key.replace('linear1', 'ffn.layers.0.0')
        new_key = new_key.replace('linear2', 'ffn.layers.1')
        new_key = new_key.replace('norm', 'norms.2')
    elif 'decoder_norm' in model_key:
        new_key = model_key.replace('sem_seg_head.predictor', 'panoptic_head')
        if 'weight' in model_key:
            state_dict[
                'panoptic_head.transformer_decoder.post_norm.weight'] = model_weight
        elif 'bias' in model_key:
            state_dict[
                'panoptic_head.transformer_decoder.post_norm.bias'] = model_weight
    elif 'mask_embed' in model_key:
        new_key = model_key.replace('sem_seg_head.predictor', 'panoptic_head')
        new_key = new_key.replace('layers.0', '0')
        new_key = new_key.replace('layers.1', '2')
        new_key = new_key.replace('layers.2', '4')
    elif 'query_embed' in model_key:
        new_key = model_key.replace('sem_seg_head.predictor', 'panoptic_head')
    elif 'level_embed' in model_key:
        new_key = model_key.replace('sem_seg_head.predictor', 'panoptic_head')
    elif 'class_input_proj' in model_key:
        new_key = model_key.replace('sem_seg_head.predictor', 'panoptic_head')
    elif 'class_embed' in model_key:
        new_key = model_key.replace('sem_seg_head.predictor.class_embed',
                                    'panoptic_head.cls_embed')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_swin(model_key, model_weight, state_dict, converted_names):

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

    if 'layers' in model_key:
        new_v = model_weight
        if 'attn.' in model_key:
            new_k = model_key.replace('attn.', 'attn.w_msa.')
        elif 'mlp.' in model_key:
            if 'mlp.fc1.' in model_key:
                new_k = model_key.replace('mlp.fc1.', 'ffn.layers.0.0.')
            elif 'mlp.fc2.' in model_key:
                new_k = model_key.replace('mlp.fc2.', 'ffn.layers.1.')
            else:
                new_k = model_key.replace('mlp.', 'ffn.')
        elif 'downsample' in model_key:
            new_k = model_key
            if 'reduction.' in model_key:
                new_v = correct_unfold_reduction_order(model_weight)
            elif 'norm.' in model_key:
                new_v = correct_unfold_norm_order(model_weight)
        else:
            new_k = model_key
        new_k = new_k.replace('layers', 'stages', 1)
    elif 'patch_embed' in model_key:
        new_v = model_weight
        if 'proj' in model_key:
            new_k = model_key.replace('proj', 'projection')
        else:
            new_k = model_key
    else:
        new_v = model_weight
        new_k = model_key

    state_dict[new_k] = new_v
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_k}')


def convert(src, dst):
    """Convert keys in detectron pretrained ResNet models to pytorch style."""
    # load arch_settings
    # load caffe model
    oneformer_model = torch.load(src)
    blobs = oneformer_model['model']
    # convert to pytorch style
    state_dict = OrderedDict()
    converted_names = set()
    for key, weight in blobs.items():
        if 'backbone' in key:
            convert_swin(key, weight, state_dict, converted_names)
        elif 'pixel_decoder' in key:
            convert_pixel_decoder(key, weight, state_dict, converted_names)
        elif 'class_transformer' in key:
            convert_class_transformer(key, weight, state_dict, converted_names)
        elif 'transformer_ffn_layers' in key or \
                'transformer_self_attention_layers' in key or \
                'transformer_cross_attention_layers' in key or \
                'sem_seg_head.predictor.decoder_norm' in key or \
                'sem_seg_head.predictor.mask_embed' in key or\
                'sem_seg_head.predictor.query_embed' in key or\
                'sem_seg_head.predictor.level_embed' in key or\
                'sem_seg_head.predictor.class_input_proj' in key or\
                'sem_seg_head.predictor.class_embed' in key:
            convert_transformer_decoder(key, weight, state_dict,
                                        converted_names)
        elif 'task_mlp' in key:
            convert_taskmlp(key, weight, state_dict, converted_names)
        elif 'text_encoder' in key:
            convert_text_encoder(key, weight, state_dict, converted_names)
        elif 'text_projector' in key:
            convert_text_projector(key, weight, state_dict, converted_names)
        elif 'prompt_ctx' in key:
            convert_prompt_ctx(key, weight, state_dict, converted_names)
    # check if all layers are converted
    for key in blobs:
        if key not in converted_names:
            print(f'Not Convert: {key}')
    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


def main():
    src = '/home/bingxing2/gpuuser206/OneFormer/checkpoints/150_16_swin_l_oneformer_coco_100ep.pth'
    dst = './150_16_swin_l_oneformer_coco_100ep.pth'
    convert(src, dst)


if __name__ == '__main__':
    main()
