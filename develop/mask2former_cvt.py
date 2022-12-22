import json
from collections import OrderedDict

import torch

from mmengine.config import Config
from mmdet.models import build_detector
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)


def get_new_name(old_name: str):
    new_name = old_name

    if 'encoder.layers' in new_name:
        new_name = new_name.replace('attentions.0', 'self_attn')

    new_name = new_name.replace('ffns.0', 'ffn')

    if 'decoder.layers' in new_name:
        new_name = new_name.replace('attentions.0', 'cross_attn')
        new_name = new_name.replace('attentions.1', 'self_attn')

    return new_name


def cvt_sd(old_sd: OrderedDict):
    new_sd = OrderedDict()
    for name, param in old_sd.items():
        new_name = get_new_name(name)
        assert new_name not in new_sd
        new_sd[new_name] = param
    assert len(new_sd) == len(old_sd)
    return new_sd


if __name__ == '__main__':

    # CFG_FILE = 'configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py'
    CFG_FILE = 'configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'
    # OLD_CKPT_FILENAME = 'mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth'
    OLD_CKPT_FILENAME = 'mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth'
    OLD_CKPT_FILEPATH = 'develop/' + OLD_CKPT_FILENAME
    NEW_CKPT_FILEPATH = 'develop/new_' + OLD_CKPT_FILENAME

    cfg = Config.fromfile(CFG_FILE)
    model_cfg = cfg.model

    detector = build_detector(model_cfg)

    refer_sd = detector.state_dict()
    old_sd = torch.load(OLD_CKPT_FILEPATH)['state_dict']

    new_sd = cvt_sd(old_sd)

    new_names = sorted(list(refer_sd.keys()))
    cvt_names = sorted(list(new_sd.keys()))
    old_names = sorted(list(old_sd.keys()))

    # we should make cvt_names --> new_names
    json.dump(new_names, open(r'./develop/new_names.json', 'w'), indent='\n')
    json.dump(cvt_names, open(r'./develop/cvt_names.json', 'w'), indent='\n')
    json.dump(old_names, open(r'./develop/old_names.json', 'w'), indent='\n')

    new_ckpt = dict(state_dict=new_sd)
    torch.save(new_ckpt, NEW_CKPT_FILEPATH)
    print(f'{NEW_CKPT_FILEPATH} has been saved!')