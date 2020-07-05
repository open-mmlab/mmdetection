from mmcv.utils import Registry, build_from_cfg

IOU_CALCULATORS = Registry('IoU calculator')


def build_iou_calculator(cfg, default_args=None):
    """Builder of IoU calculator."""
    return build_from_cfg(cfg, IOU_CALCULATORS, default_args)
