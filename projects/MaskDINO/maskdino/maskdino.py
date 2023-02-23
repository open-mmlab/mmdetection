# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.models import SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class MaskDINO(SingleStageDetector):
    r"""Implementation of `Mask DINO: Towards A Unified Transformer-based
    Framework for Object Detection and Segmentation
    <https://arxiv.org/abs/2206.02777>`_."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 panoptic_head: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        panoptic_head_ = panoptic_head.deepcopy()
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = MODELS.build(panoptic_head_)

        # self.num_things_classes = self.panoptic_head.num_things_classes
        # self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        x = self.extract_feat(batch_inputs)
        losses = self.panoptic_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:  # rescale not used
        feats = self.extract_feat(batch_inputs)
        results = self.panoptic_head.predict(feats, batch_data_samples)
        return results

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        raise NotImplementedError()
        # feats = self.extract_feat(batch_inputs)
        # results = self.panoptic_head.forward(feats, batch_data_samples)
        # return results
