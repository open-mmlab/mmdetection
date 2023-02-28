# script of converting detr ckpt of mmdet-2.x to mmdet-3.x
import json
from pathlib import Path
from typing import OrderedDict
import torch
from mmengine import Config, build_model_from_cfg
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

register_all_modules()

INDEX = 0


def get_mapped_name(name: str):
    new_name = name

    if 'backbone' in new_name:
        new_name = convert_backbone_keys(new_name)

    elif 'sem_seg_head' in new_name:
        new_name = new_name.replace('sem_seg_head', 'panoptic_head')
        # bbox_embed
        new_name = new_name.replace('decoder.bbox_embed', 'bbox_embed')
        # transformer encoder
        if 'transformer.encoder' in new_name:
            new_name = new_name.replace('linear1', 'ffn.layers.0.0')
            new_name = new_name.replace('linear2', 'ffn.layers.1')
            new_name = new_name.replace('norm1', 'norms.0')
            new_name = new_name.replace('norm2', 'norms.1')
        # conv in pixel decoder
        if ('pixel_decoder.adapter_1' in new_name
                or 'pixel_decoder.layer_1' in new_name):
            new_name = new_name.replace('norm', 'gn')
            new_name = new_name.replace('_1.weight', '_1.conv.weight')

    elif 'criterion' in new_name:
        new_name = new_name.replace('criterion', 'panoptic_head.criterion')

    return new_name


def convert_backbone_keys(name: str):
    if 'backbone.stem' in name:
        name = name.replace('backbone.stem.conv1.weight', 'backbone.conv1.weight')
        name = name.replace('backbone.stem.conv1.norm', 'backbone.bn1')
    elif 'backbone.res' in name:
        key_name_split = name.split('.')
        # weight_type = key_name_split[-1]
        res_id = int(key_name_split[1][-1]) - 1
        # deal with short cut
        if 'shortcut' in key_name_split[3]:
            if 'shortcut' == key_name_split[-2]:
                name = f'backbone.layer{res_id}.' \
                       f'{key_name_split[2]}.downsample.0.' \
                       f'{key_name_split[-1]}'
            elif 'shortcut' == key_name_split[-3]:
                name = f'backbone.layer{res_id}.' \
                       f'{key_name_split[2]}.downsample.1.' \
                       f'{key_name_split[-1]}'
            else:
                print(f'Unvalid key {k}')
        # deal with conv
        elif 'conv' in key_name_split[-2]:
            conv_id = int(key_name_split[-2][-1])
            name = f'backbone.layer{res_id}.{key_name_split[2]}' \
                   f'.conv{conv_id}.{key_name_split[-1]}'
        # deal with BN
        elif key_name_split[-2] == 'norm':
            conv_id = int(key_name_split[-3][-1])
            name = f'backbone.layer{res_id}.{key_name_split[2]}.' \
                   f'bn{conv_id}.{key_name_split[-1]}'
        else:
            print(f'{k} is invalid')
    return name


def mapping_state_dict(state_dict: OrderedDict) -> OrderedDict:
    out = OrderedDict()
    for name, param in state_dict.items():
        new_name = get_mapped_name(name)
        assert new_name not in out, f'{name}-->{new_name}'
        out[new_name] = param
    return out


def preprocess_state_dict(state_dict: OrderedDict) -> OrderedDict:
    out = delete_duplicated_items(state_dict)
    return out


def delete_duplicated_items(state_dict: OrderedDict) -> OrderedDict:
    out = OrderedDict()
    for name, param in state_dict.items():
        if 'decoder_norm' in name:
            continue
        out[name] = param
    return out


if __name__ == '__main__':
    cfg_3x = Path(['configs/maskdino_r50_8xb2-lsj-50e_coco-panoptic.py',][INDEX])
    ckpt_2x = Path(['model_converters/'
                    'maskdino_r50_50ep_300q_hid2048_3sd1_panoptic_pq53.0.pth',][INDEX])
    save_path = str(ckpt_2x.parent) + f'/aligned_{ckpt_2x.name}'

    cfg_3x = Config.fromfile(cfg_3x)
    model_cfg = cfg_3x.model.copy()
    model = build_model_from_cfg(model_cfg, MODELS)

    sd_3x = model.state_dict()
    ckpt_2x = torch.load(ckpt_2x)
    sd_2x = ckpt_2x.pop('model')

    # convert
    sd_2x = preprocess_state_dict(sd_2x)
    sd_2x = mapping_state_dict(sd_2x)

    name_3x = sorted(list(sd_3x.keys()))
    name_2x = sorted(list(sd_2x.keys()))
    json.dump(name_3x, open(
        r'./model_converters/names_1.json', 'w'), indent=0)
    json.dump(name_2x, open(
        r'./model_converters/names_2.json', 'w'), indent=0)

    torch.save(sd_2x, save_path)