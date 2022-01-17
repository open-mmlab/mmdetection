# Copyright (c) OpenMMLab. All rights reserved.
try:
    import timm
except ImportError:
    timm = None

import warnings

from mmcv.cnn.bricks.registry import NORM_LAYERS
from mmcv.runner import BaseModule

from ...utils import get_root_logger
from ..builder import BACKBONES


@BACKBONES.register_module()
class TIMMBackbone(BaseModule):
    """Wrapper to use backbones from timm library.

    More details can be found in
    `timm <https://github.com/rwightman/pytorch-image-models>`_.
    See especially the document for `feature extraction
    <https://rwightman.github.io/pytorch-image-models/feature_extraction/>`_.

    Args:
        model_name (str): Name of timm model to instantiate.
        features_only (bool, optional): Whether to create backbone for
            extracting feature maps from the deepest layer at each stride.
            For typical use of CNNs, set this True. For Vision Transformer
            models that do not support this argument, set this False.
            Default: True.
        pretrained (bool, optional): Whether to load pretrained weights.
            Default: True.
        checkpoint_path (str, optional): Path of checkpoint to load at the
            last of timm.create_model. Default: '', which means not loading.
        in_channels (int, optional): Number of input image channels.
            Default: 3.
        init_cfg (dict or list[dict], optional): Initialization config dict of
            OpenMMLab projects. Default: None.
        **kwargs: Other timm & model specific arguments.
    """

    def __init__(self,
                 model_name,
                 features_only=True,
                 pretrained=True,
                 checkpoint_path='',
                 in_channels=3,
                 init_cfg=None,
                 **kwargs):
        if timm is None:
            raise RuntimeError(
                'Failed to import timm. Please run "pip install timm". '
                '"pip install dataclasses" may also be needed for Python 3.6.')
        if not isinstance(pretrained, bool):
            raise TypeError('pretrained must be bool, not str for model path')
        if features_only and checkpoint_path:
            warnings.warn(
                'Using both features_only and checkpoint_path will cause error'
                ' in timm. See '
                'https://github.com/rwightman/pytorch-image-models/issues/488')

        super(TIMMBackbone, self).__init__(init_cfg)
        if 'norm_layer' in kwargs:
            kwargs['norm_layer'] = NORM_LAYERS.get(kwargs['norm_layer'])
        self.timm_model = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            **kwargs)

        # reset classifier
        if hasattr(self.timm_model, 'reset_classifier'):
            self.timm_model.reset_classifier(0, '')

        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True

        feature_info = self.timm_model.feature_info
        logger = get_root_logger()
        logger.info(f'backbone out_indices: {feature_info.out_indices}')
        logger.info(f'backbone out_channels: {feature_info.channels()}')
        logger.info(f'backbone out_strides: {feature_info.reduction()}')

    def forward(self, x):
        features = self.timm_model(x)
        return features
