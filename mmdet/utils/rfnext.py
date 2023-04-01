# Copyright (c) OpenMMLab. All rights reserved.
try:
    from mmcv.cnn import RFSearchHook
except ImportError:
    RFSearchHook = None


def rfnext_init_model(detector, cfg):
    """Rcecptive field search via dilation rates.

    Please refer to `RF-Next: Efficient Receptive Field
    Search for Convolutional Neural Networks
    <https://arxiv.org/abs/2206.06637>`_ for more details.

    Args:
        detector (nn.Module): The detector before initializing RF-Next.
        cfg (mmcv.Config): The config for RF-Next.
            If the RFSearchHook is defined in the cfg.custom_hooks,
            the detector will be initialized for RF-Next.
    """

    if cfg.get('custom_hooks', None) is None:
        return
    custom_hook_types = [hook['type'] for hook in cfg.custom_hooks]
    if 'RFSearchHook' not in custom_hook_types:
        return

    index = custom_hook_types.index('RFSearchHook')
    rfsearch_cfg = cfg.custom_hooks[index]
    assert rfsearch_cfg['type'] == 'RFSearchHook'

    assert RFSearchHook is not None, 'Please install mmcv > 1.7.0'

    # initlize a RFSearchHook
    rfsearch_warp = RFSearchHook(
        mode=rfsearch_cfg.get('mode', 'search'),
        config=rfsearch_cfg.get('config', None),
        rfstructure_file=rfsearch_cfg.get('rfstructure_file', None),
        by_epoch=rfsearch_cfg.get('by_epoch', True),
        verbose=rfsearch_cfg.get('verbose', True),
    )
    rfsearch_warp.init_model(detector)
    rfsearch_cfg['rfstructure_file'] = None
