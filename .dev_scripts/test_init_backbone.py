"""Check out backbone whether successfully load pretrained checkpoint."""
import copy
import os
from os.path import dirname, exists, join

import pytest
from mmcv import Config
from mmcv.runner import _load_checkpoint

from mmdet.models import build_detector


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(__file__))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet
        repo_dpath = dirname(dirname(mmdet.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


def _traversed_config_file():
    """Traversed all potential config files under the `config` file."""
    config_path = _get_config_directory()
    check_cfg_names = []

    # `base`, `legacy_1.x` and `common` ignored by default.
    ignore_cfg_names = ['_base_', 'legacy_1.x', 'common']
    # 'ld' need load teacher model, if want to check 'ld',
    # please check teacher_config path first.
    ignore_cfg_names += ['ld']
    # `selfsup_pretrain` need convert model, if want to check this model,
    # need to convert the model first.
    ignore_cfg_names += ['selfsup_pretrain']

    # the `init_cfg` in 'centripetalnet', 'cornernet', 'cityscapes' and
    # 'scratch' is None.
    # Please confirm `bockbone.init_cfg` is None first.
    ignore_cfg_names += [
        'centripetalnet', 'cornernet', 'cityscapes', 'scratch'
    ]

    for config_file_name in os.listdir(config_path):
        if config_file_name in ignore_cfg_names:
            continue
        config_file = join(config_path, config_file_name)
        if os.path.isdir(config_file):
            for config_sub_file in os.listdir(config_file):
                if config_sub_file.endswith('py'):
                    name = join(config_file, config_sub_file)
                    check_cfg_names.append(name)
    return check_cfg_names


def _check_backbone(config, print_cfg=True):
    """Check out backbone whether successfully load pretrained model, by using
    `backbone.init_cfg`.

    First, using `mmcv._load_checkpoint` to load the checkpoint without
        loading models.
    Then, using `build_detector` to build models, and using
        `model.init_weights()` to initialize the parameters.
    Finally, assert weights and bias of each layer loaded from pretrained
        checkpoint are equal to the weights and bias of original checkpoint.
        For the convenience of comparison, we sum up weights and bias of
        each loaded layer separately.

    Args:
        config (str): Config file path.
        print_cfg (bool): Whether print logger and return the result.

    Returns:
        results (str or None): If backbone successfully load pretrained
            checkpoint, return None; else, return config file path.
    """
    if print_cfg:
        print('-' * 15 + 'loading ', config)
    cfg = Config.fromfile(config)
    init_cfg = None
    try:
        init_cfg = cfg.model.backbone.init_cfg
        init_flag = True
    except AttributeError:
        init_flag = False
    if init_cfg is None or init_cfg.get('type') != 'Pretrained':
        init_flag = False
    if init_flag:
        checkpoint = _load_checkpoint(init_cfg.checkpoint)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        model.init_weights()

        checkpoint_layers = dict()
        backbone_layers = dict()
        for name, value in state_dict.items():
            split_name = name.split('.')
            if split_name[-1] == 'weight':
                checkpoint_layers[name] = value.sum()
        for name, value in model.backbone.state_dict().items():
            split_name = name.split('.')
            if split_name[-1] == 'weight':
                backbone_layers[name] = value.sum()

        check_layers = list(set(backbone_layers) & set(checkpoint_layers))
        for layer_name in check_layers:
            assert backbone_layers[layer_name] == checkpoint_layers[layer_name]

        if print_cfg:
            print('-' * 10 + 'Successfully load checkpoint' + '-' * 10 +
                  '\n', )
            return None
    else:
        if print_cfg:
            print(config + '\n' + '-' * 10 +
                  'config file do not have init_cfg' + '-' * 10 + '\n')
            return config


check_cfg_names = _traversed_config_file()


@pytest.mark.parametrize('config', check_cfg_names)
def test_load_pretrained(config):
    """Check out backbone whether successfully load pretrained model by using
    `backbone.init_cfg`.

    Details please refer to `_check_backbone`
    """
    _check_backbone(config, print_cfg=False)


def _test_load_pretrained():
    """We traversed all potential config files under the `config` file. If you
    need to print details or debug code, you can use this function.

    Returns
        check_cfg_names (list[str]): Config files that backbone initialized
            from pretrained checkpoint might be problematic. Need to recheck
            the config file. The output including the config files that the
            backbone.init_cfg is None
    # >>> from test_init_backbone import _test_load_pretrained
    # >>> check_cfg_names = _test_load_pretrained()
    # >>> print('These config files need to be checked again')
    # >>> print(check_cfg_names)
    """
    config_path = _get_config_directory()
    check_cfg_names = []

    # `base`, `legacy_1.x` and `common` ignored by default.
    ignore_cfg_names = ['_base_', 'legacy_1.x', 'common']
    # 'ld' need load teacher model, if want to check 'ld',
    # please check teacher_config path first.
    ignore_cfg_names += ['ld']
    # `selfsup_pretrain` need convert model, if want to check this model,
    # need to convert the model first.
    ignore_cfg_names += ['selfsup_pretrain']

    # the `init_cfg` in 'centripetalnet', 'cornernet', 'cityscapes',
    # 'scratch' is None.
    # the `init_cfg` in ssdlite(`ssdlite_mobilenetv2_scratch_600e_coco.py`)
    # is None
    #  Please confirm `bockbone.init_cfg` is None first.
    ignores_folder = ['centripetalnet', 'cornernet', 'cityscapes', 'scratch']
    ignores_file = ['ssdlite_mobilenetv2_scratch_600e_coco.py']
    for config_file_name in os.listdir(config_path):
        if config_file_name in ignore_cfg_names:
            continue
        config_file = join(config_path, config_file_name)
        if os.path.isdir(config_file):
            for config_sub_file in os.listdir(config_file):
                if config_sub_file.endswith('py'):
                    name = join(config_file, config_sub_file)
                    init_cfg_name = _check_backbone(name)
                    if init_cfg_name is not None:
                        # ignore config files that `init_cfg` is None
                        if config_file_name not in ignores_folder and \
                                config_sub_file not in ignores_file:
                            check_cfg_names.append(name)
    return check_cfg_names
