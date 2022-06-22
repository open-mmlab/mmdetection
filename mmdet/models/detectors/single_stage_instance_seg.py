# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple

from torch import Tensor

from mmdet.core.utils import (ConfigType, OptConfigType, OptMultiConfig,
                              SampleList)
from mmdet.registry import MODELS
from .base import BaseDetector

INF = 1e8


@MODELS.register_module()
class SingleStageInstanceSegmentor(BaseDetector):
    """Base class for single-stage instance segmentors."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 mask_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        else:
            self.neck = None
        if bbox_head is not None:
            bbox_head.update(train_cfg=copy.deepcopy(train_cfg))
            bbox_head.update(test_cfg=copy.deepcopy(test_cfg))
            self.bbox_head = MODELS.build(bbox_head)
        else:
            self.bbox_head = None

        assert mask_head, f'`mask_head` must ' \
                          f'be implemented in {self.__class__.__name__}'
        mask_head.update(train_cfg=copy.deepcopy(train_cfg))
        mask_head.update(test_cfg=copy.deepcopy(test_cfg))
        self.mask_head = MODELS.build(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have different
            resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _forward(self, batch_inputs: Tensor, *args, **kwargs) \
            -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        outs = ()
        # backbone
        x = self.extract_feat(batch_inputs)
        # bbox_head
        if self.with_bbox:
            # TODO: current not supported
            pass
        # mask_head
        mask_outs = self.mask_head.forward(x)
        outs = outs + (mask_outs, )
        return outs

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList,
             **kwargs) -> dict:
        """
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
        losses = dict()

        # TODO: Check the logic in CondInst and YOLACT
        positive_infos = None
        # CondInst and YOLACT have bbox_head
        if self.with_bbox:
            bbox_losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
            # TODO: enhance the logic when refactor YOLACT
            if bbox_losses.get('positive_infos', None) is not None:
                positive_infos = bbox_losses.pop('positive_infos')
            losses.update(bbox_losses)

        mask_loss = self.mask_head.loss(
            x, batch_data_samples, positive_infos=positive_infos, **kwargs)
        # avoid loss override
        assert not set(mask_loss.keys()) & set(losses.keys())

        losses.update(mask_loss)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False,
                **kwargs) -> SampleList:
        """Perform forward propagation of the mask head and predict mask
        results on the features of the upstream network.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        x = self.extract_feat(batch_inputs)
        if self.with_bbox:
            # TODO: currently not checked
            bbox_results_list = self.bbox_head.predict(
                x, batch_data_samples, rescale=rescale)
        else:
            bbox_results_list = None

        results_list = self.mask_head.predict(
            x,
            batch_data_samples,
            rescale=rescale,
            bbox_results_list=bbox_results_list)

        # connvert to DetDataSample
        results_list = self.convert_to_datasample(results_list)
        return results_list
