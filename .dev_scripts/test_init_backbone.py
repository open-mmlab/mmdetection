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
    except Warning:
        init_flag = False
    if init_cfg is None:
        init_flag = False
    if init_flag:
        checkpoint = _load_checkpoint(init_cfg.checkpoint)
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        model.init_weights()

        layers = []
        names = []
        checkpoint_names = []
        layer_indexes = []
        darknet_indexs = []
        for name in model.backbone.state_dict():
            split_name = name.split('.')
            if split_name[-1] == 'weight':
                if 'stem' in split_name[0]:
                    layers.append(split_name[0])
                    layer_indexes.append(split_name[1])
                    checkpoint_names.append(name)
                    names.append(None)
                    darknet_indexs.append(None)
                if 'features' in split_name[0]:
                    names.append(None)
                    checkpoint_names.append(name)
                    layer_indexes.append(split_name[1])
                    layers.append(split_name[0])
                    darknet_indexs.append(None)
                if 'conv' in split_name[-2]:
                    # conv
                    if len(split_name) == 2:
                        # conv in backbone
                        names.append(split_name[-2])
                        checkpoint_names.append(name)
                        layer_indexes.append(None)
                        layers.append(None)
                        darknet_indexs.append(None)
                    elif len(split_name) == 3:
                        # darknet
                        names.append(split_name[-2])
                        checkpoint_names.append(name)
                        layer_indexes.append(None)
                        layers.append(split_name[0])
                        darknet_indexs.append(None)
                        # vgg
                    elif len(split_name) == 4:
                        if 'layer' in split_name[0]:
                            # resnet
                            names.append(split_name[-2])
                            checkpoint_names.append(name)
                            layer_indexes.append(split_name[1])
                            layers.append(split_name[0])
                            darknet_indexs.append(None)
                    elif len(split_name) == 5:
                        if 'conv_res' in split_name[0]:
                            names.append(split_name[-2])
                            checkpoint_names.append(name)
                            layer_indexes.append(split_name[2])
                            layers.append(split_name[0])
                            darknet_indexs.append(split_name[1])

        assert len(names) > 0
        assert len(names) == len(checkpoint_names) == \
               len(layer_indexes) == len(layers) == len(darknet_indexs)
        for i in range(len(names)):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            weight_sum = state_dict[checkpoint_names[i]].sum()
            if layer_indexes[i] is None:
                if layers[i] is None:
                    after_init_weight_sum = getattr(model.backbone,
                                                    names[i]).weight.sum()
                else:
                    layer = getattr(model.backbone, layers[i])
                    after_init_weight_sum = getattr(layer,
                                                    names[i]).weight.sum()
            else:
                if darknet_indexs[i] is not None:
                    layer = getattr(model.backbone, layers[i])
                    conv = getattr(layer, darknet_indexs[i])
                    conv = getattr(conv, layer_indexes[i])
                    after_init_weight_sum = getattr(conv,
                                                    names[i]).weight.sum()
                else:
                    if names[i] is None:
                        conv = getattr(model.backbone,
                                       layers[i])[int(layer_indexes[i])]
                        after_init_weight_sum = conv.weight.sum()
                    else:
                        layer = getattr(model.backbone,
                                        layers[i])[int(layer_indexes[i])]
                        after_init_weight_sum = getattr(layer,
                                                        names[i]).weight.sum()
            assert weight_sum == after_init_weight_sum
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
    _check_backbone(config, print_cfg=True)


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

    #  the `init_cfg` in 'centripetalnet', 'cornernet', 'cityscapes' and
    #  'scratch' is None.
    #  Please confirm `bockbone.init_cfg` is None first.
    ignores = ['centripetalnet', 'cornernet', 'cityscapes', 'scratch']

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
                        if config_file_name not in ignores:
                            check_cfg_names.append(name)
    return check_cfg_names
