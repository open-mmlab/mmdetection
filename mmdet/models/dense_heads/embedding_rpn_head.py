# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.structures.det_data_sample import SampleList
from mmdet.utils import InstanceList, OptConfigType


@MODELS.register_module()
class EmbeddingRPNHead(BaseModule):
    """RPNHead in the `Sparse R-CNN <https://arxiv.org/abs/2011.12450>`_ .

    Unlike traditional RPNHead, this module does not need FPN input, but just
    decode `init_proposal_bboxes` and expand the first dimension of
    `init_proposal_bboxes` and `init_proposal_features` to the batch_size.

    Args:
        num_proposals (int): Number of init_proposals. Defaults to 100.
        proposal_feature_channel (int): Channel number of
            init_proposal_feature. Defaults to 256.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 num_proposals: int = 100,
                 proposal_feature_channel: int = 256,
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        # `**kwargs` is necessary to avoid some potential error.
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)
        self.num_proposals = num_proposals
        self.proposal_feature_channel = proposal_feature_channel
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize a sparse set of proposal boxes and proposal features."""
        self.init_proposal_bboxes = nn.Embedding(self.num_proposals, 4)
        self.init_proposal_features = nn.Embedding(
            self.num_proposals, self.proposal_feature_channel)

    def init_weights(self) -> None:
        """Initialize the init_proposal_bboxes as normalized.

        [c_x, c_y, w, h], and we initialize it to the size of  the entire
        image.
        """
        super().init_weights()
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)

    def _decode_init_proposals(self, x: List[Tensor],
                               batch_data_samples: SampleList) -> InstanceList:
        """Decode init_proposal_bboxes according to the size of images and
        expand dimension of init_proposal_features to batch_size.

        Args:
            x (list[Tensor]): List of FPN features.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            List[:obj:`InstanceData`:] Detection results of each image.
            Each item usually contains following keys.

            - proposals: Decoded proposal bboxes,
              has shape (num_proposals, 4).
            - features: init_proposal_features, expanded proposal
              features, has shape
              (num_proposals, proposal_feature_channel).
            - imgs_whwh: Tensor with shape
              (num_proposals, 4), the dimension means
              [img_width, img_height, img_width, img_height].
        """
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)

        proposals = self.init_proposal_bboxes.weight.clone()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        imgs_whwh = []
        for meta in batch_img_metas:
            h, w = meta['img_shape'][:2]
            imgs_whwh.append(x[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]
        proposals = proposals * imgs_whwh

        rpn_results_list = []
        for idx in range(len(batch_img_metas)):
            rpn_results = InstanceData()
            rpn_results.bboxes = proposals[idx]
            rpn_results.imgs_whwh = imgs_whwh[idx].repeat(
                self.num_proposals, 1)
            rpn_results.features = self.init_proposal_features.weight.clone()
            rpn_results_list.append(rpn_results)
        return rpn_results_list

    def loss(self, *args, **kwargs):
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network."""
        raise NotImplementedError(
            'EmbeddingRPNHead does not have `loss`, please use '
            '`predict` or `loss_and_predict` instead.')

    def predict(self, x: List[Tensor], batch_data_samples: SampleList,
                **kwargs) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network."""
        # `**kwargs` is necessary to avoid some potential error.
        return self._decode_init_proposals(
            x=x, batch_data_samples=batch_data_samples)

    def loss_and_predict(self, x: List[Tensor], batch_data_samples: SampleList,
                         **kwargs) -> tuple:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples."""
        # `**kwargs` is necessary to avoid some potential error.
        predictions = self._decode_init_proposals(
            x=x, batch_data_samples=batch_data_samples)

        return dict(), predictions
