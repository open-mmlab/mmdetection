from mmdet.utils import build_from_cfg
from .registry import ANCHOR_GENERATORS


def build_anchor_generator(cfg, default_args=None):
    return build_from_cfg(cfg, ANCHOR_GENERATORS, default_args)
