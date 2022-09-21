# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from ..layers import MLP, inverse_sigmoid
from .detr_head import DETRHead


@MODELS.register_module()
class DABDETRHead(DETRHead):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        
    """

    def __init__(
            self,
            *args,
            # bbox_embed_diff_each_layer=False,
            **kwargs) -> None:
        # self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the transformer head."""
        # cls branch
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        # reg branch
        self.fc_reg = MLP(self.embed_dims, self.embed_dims, 4, 3)
        # NOTE the activations of reg_branch is the same as those in
        # transformer, but they are actually different in Conditional DETR
        # and DAB DETR (prelu in transformer and relu in reg_branch)

    # def _load_from_state_dict  # TODO

    def init_weights(self) -> None:
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
        constant_init(self.fc_reg.layers[-1], 0., bias=0.)

    def forward(self, outs_dec: Tensor,
                reference: Tensor) -> Tuple[Tensor, Tensor]:
        """"Forward function for a single feature level.

        Args: TODO

        Returns:
            tuple[Tensor]:

            - all_cls_scores (Tensor): Outputs from the classification head, \
            shape [nb_dec, bs, num_query, cls_out_channels]. Note \
            cls_out_channels should includes background.
            - all_bbox_preds (Tensor): Sigmoid outputs from the regression \
            head with normalized coordinate format (cx, cy, w, h). \
            Shape [nb_dec, bs, num_query, 4].
        """
        all_cls_scores = self.fc_cls(outs_dec)
        reference_before_sigmoid = inverse_sigmoid(reference, eps=1e-3)
        tmp = self.fc_reg(outs_dec)
        tmp[..., :reference_before_sigmoid.size(-1
                                                )] += reference_before_sigmoid
        all_bbox_preds = tmp.sigmoid()
        return all_cls_scores, all_bbox_preds

    def loss(self, outs_dec: Tensor, reference: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Outputs from the transformer detector, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(outs_dec, reference)
        loss_inputs = outs + (batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def predict(self,
                outs_dec: Tensor,
                reference: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network. Over-write
        because img_metas are needed as inputs for bbox_head.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(outs_dec, reference)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions
