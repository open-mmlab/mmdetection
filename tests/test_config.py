from os.path import join, dirname, exists


def _get_config_directory():
    """ Find the predefined detector config directory """
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


def test_config_build_detector():
    """
    Test that all detection models defined in the configs can be initialized.
    """
    import glob
    from xdoctest.utils import import_module_from_path
    from mmdet.models import build_detector

    config_dpath = _get_config_directory()
    print('Found config_dpath = {!r}'.format(config_dpath))

    config_fpaths = list(glob.glob(join(config_dpath, '**', '*.py')))
    print('Found {} config files'.format(len(config_fpaths)))

    for config_fpath in config_fpaths:
        config_mod = import_module_from_path(config_fpath)

        config_mod.model
        config_mod.train_cfg
        config_mod.test_cfg
        print('Building detector from config_fpath = {!r}'.format(config_fpath))

        # Remove pretrained keys to allow for testing in an offline environment
        if 'pretrained' in config_mod.model:
            config_mod.model['pretrained'] = None

        detector = build_detector(config_mod.model,
                                  train_cfg=config_mod.train_cfg,
                                  test_cfg=config_mod.test_cfg)
        assert detector is not None
