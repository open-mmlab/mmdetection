import pickle as pkl
from collections import OrderedDict

import torch

# weight = torch.load('/home/yhcao/tmp/state_dict.pth')
weight = torch.load('data/FCOS_R_50_FPN_1x.pth')['model']
new_weight = OrderedDict()

for k, v in weight.items():
    if 'body' in k:
        k = k.replace('body.', '')
    if 'stem' in k:
        k = k.replace('stem.', '')
    if 'fpn' in k:
        k = k.replace('backbone.fpn', 'neck')
        if 'fpn_inner2' in k:
            k = k.replace('fpn_inner2', 'lateral_convs.0.conv')
        if 'fpn_inner3' in k:
            k = k.replace('fpn_inner3', 'lateral_convs.1.conv')
        if 'fpn_inner4' in k:
            k = k.replace('fpn_inner4', 'lateral_convs.2.conv')
        if 'fpn_layer2' in k:
            k = k.replace('fpn_layer2', 'fpn_convs.0.conv')
        if 'fpn_layer3' in k:
            k = k.replace('fpn_layer3', 'fpn_convs.1.conv')
        if 'fpn_layer4' in k:
            k = k.replace('fpn_layer4', 'fpn_convs.2.conv')
        if 'top_blocks.p6' in k:
            k = k.replace('top_blocks.p6', 'fpn_convs.3.conv')
        if 'top_blocks.p7' in k:
            k = k.replace('top_blocks.p7', 'fpn_convs.4.conv')
    if 'rpn.head' in k:
        k = k.replace('rpn.head', 'bbox_head')
        if 'cls_tower' in k:
            k = k.replace('cls_tower', 'cls_layers')
        if 'bbox_tower' in k:
            k = k.replace('bbox_tower', 'reg_layers')
        if 'cls_logits' in k:
            k = k.replace('cls_logits', 'fcos_cls')
        if 'bbox_pred' in k:
            k = k.replace('bbox_pred', 'fcos_reg')
        if 'centerness' in k:
            k = k.replace('centerness', 'fcos_centerness')
    if 'box_selector_test' in k:
        continue
    new_weight[k] = v

for k, v in new_weight.items():
    print(k, v.sum().item())
new_weight = dict(state_dict=new_weight)
torch.save(new_weight, open('data/tmp.pth', 'wb'))
