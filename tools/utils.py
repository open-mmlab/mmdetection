# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
from mmcv.utils import print_log

def update_data_root(cfg, data_root_n, logger=None):
    """updata data root

    Args:
        cfg (mmcv.Config): model config
        data_root_n (str): new data root
        logger (logging.Logger | str | None): the way to print msg

    """
    assert isinstance(cfg, mmcv.Config), \
        f"cfg got wrong type: {type(cfg)}, expected mmcv.Config"
    
    def update(cfg, str_o, str_n):
        for k, v in cfg.items():
            if isinstance(v, mmcv.ConfigDict):
                update(cfg[k], str_o, str_n)
            if isinstance(v, str) and str_o in v:
                cfg[k] = v.replace(str_o, str_n)

    update(cfg.data, cfg.data_root, data_root_n)
    cfg.data_root = data_root_n
    print_log(
        f"Set data root to {data_root_n} according to MMDET_DATASETS", 
        logger=logger)