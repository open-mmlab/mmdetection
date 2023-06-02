# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector


@MODELS.register_module()
class CrowdDet(TwoStageDetector):
    """Implementation of `CrowdDet <https://arxiv.org/abs/2003.09163>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        rpn_head (:obj:`ConfigDict` or dict): The rpn config.
        roi_head (:obj:`ConfigDict` or dict): The roi config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of FCOS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of FCOS. Defaults to None.
        neck (:obj:`ConfigDict` or dict): The neck config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
