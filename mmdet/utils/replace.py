# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import re

from mmcv.utils import Config


def replace_config(ori_cfg):
    """Replace the ${key} with the value of ori_cfg.key in the config. And
    support replacing the chained ${key}. Such as, replace ${key0.key1} with
    the value of cfg.key0.key1. Code is modified from `vars.py
    https://github.com/microsoft/SoftTeacher/blob/main/ssod/utils/vars.py`

    Args:
        ori_cfg(mmcv.utils.config.Config):
        The origin config with ${key} generated from a file.

    Returns:
        replaced_cfg(mmcv.utils.config.Config):
        The config with ${key} replaced by the corresponding value.
    """

    def get_value(cfg, key):
        chained_key = key.split('.')
        if len(chained_key) == 1:
            return cfg[chained_key[0]]
        else:
            return get_value(cfg[chained_key[0]], '.'.join(chained_key[1:]))

    def replace_value(cfg):
        if isinstance(cfg, dict):
            return {key: replace_value(value) for key, value in cfg.items()}
        elif isinstance(cfg, list):
            return [replace_value(item) for item in cfg]
        elif isinstance(cfg, tuple):
            return tuple([replace_value(item) for item in cfg])
        elif isinstance(cfg, str):
            # replace the ${key} with cfg.key
            keys = pattern_key.findall(cfg)
            values = [get_value(ori_cfg_dict, key[2:-1]) for key in keys]
            # only support replacing one ${key} for dict, list, or tuple
            for key, value in zip(keys, values):
                cfg = value if len(keys) == 1 else cfg.replace(key, str(value))
            return cfg
        else:
            return cfg

    # the pattern of key, which will be replaced by its value, such as ${model}
    pattern_key = re.compile(r'\$\{[a-zA-Z\d_.]*\}')
    # ori_cfg is the cfg before being replaced
    ori_cfg_dict = ori_cfg._cfg_dict.to_dict()
    # work_dirs/${cfg_name}/${percent}/${fold}
    ori_cfg_dict['cfg_name'] = osp.splitext(osp.basename(ori_cfg.filename))[0]
    replaced_cfg = Config(
        replace_value(ori_cfg_dict), filename=ori_cfg.filename)
    # replace the model with semi_wrapper
    if replaced_cfg.get('semi_wrapper', None) is not None:
        replaced_cfg.model = replaced_cfg.semi_wrapper
        replaced_cfg.pop('semi_wrapper')

    return replaced_cfg
