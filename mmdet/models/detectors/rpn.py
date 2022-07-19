# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector


@MODELS.register_module()
class RPN(SingleStageDetector):
    """Implementation of Region Proposal Network.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 rpn_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        rpn_train_cfg = train_cfg['rpn'] if train_cfg is not None else None
        rpn_head_num_classes = rpn_head.get('num_classes', 1)
        if rpn_head_num_classes != 1:
            warnings.warn('The `num_classes` should be 1 in RPN, but get '
                          f'{rpn_head_num_classes}, please set '
                          'rpn_head.num_classes = 1 in your config file.')
            rpn_head.update(num_classes=1)
        rpn_head.update(train_cfg=rpn_train_cfg)
        rpn_head.update(test_cfg=test_cfg['rpn'])
        self.bbox_head = MODELS.build(rpn_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)

        # set cat_id of gt_labels to 0 in RPN
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        losses = self.bbox_head.loss(x, rpn_data_samples)
        return losses
