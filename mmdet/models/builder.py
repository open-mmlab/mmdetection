from torch import nn

from mmdet.utils import build_from_cfg
from .registry import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                       ROI_EXTRACTORS, SHARED_HEADS)


def build(cfg, registry, default_args=None):#建立模型的所有几口函数，
  '''
  函数的作用是用于构建网络的所有模块，他会根据模型被分成多少个部分，然后根据给定的参数字典来构建相应的部分，
  如果给定的参数是一个list，那么他就生成了序列模块，不然就是用整个字典进行构建
  '''
    if isinstance(cfg, list):#如果给定的参数是列表格式，这个是模型中一小部分的构建
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]#生成模块序列列表
        return nn.Sequential(*modules)#生成序列模型
    else:
        return build_from_cfg(cfg, registry, default_args)#利用给定的字典生成模型，其中的registry是一个定义的类


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    return build(cfg, SHARED_HEADS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_detector(cfg, train_cfg=None, test_cfg=None):#函数调用接口，cfg，train_cfg,test_cfg是一个字典，
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
