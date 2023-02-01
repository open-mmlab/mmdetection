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
from .conditional_detr_head import ConditionalDETRHead


@MODELS.register_module()
class DABDETRHead(ConditionalDETRHead):
    """Head of DAB-DETR. DAB-DETR: Dynamic Anchor Boxes are Better Queries for
    DETR.

    More details can be found in the `paper
    <https://arxiv.org/abs/2201.12329>`_ .
    """

    def _init_layers(self) -> None:
        """Initialize layers of the transformer head."""
        # cls branch
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        # reg branch
        self.fc_reg = MLP(self.embed_dims, self.embed_dims, 4, 3)

    def init_weights(self) -> None:
        """initialize weights."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
        constant_init(self.fc_reg.layers[-1], 0., bias=0.)

    def forward(self, hidden_states: Tensor,
                references: Tensor) -> Tuple[Tensor, Tensor]:
        """"Forward function.

        Args:
            hidden_states (Tensor): Features from transformer decoder. If
                `return_intermediate_dec` is True output has shape
                (num_decoder_layers, bs, num_queries, dim), else has shape (1,
                bs, num_queries, dim) which only contains the last layer
                outputs.
            references (Tensor): References from transformer decoder. If
                `return_intermediate_dec` is True output has shape
                (num_decoder_layers, bs, num_queries, 2/4), else has shape (1,
                bs, num_queries, 2/4)
                which only contains the last layer reference.
        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - layers_cls_scores (Tensor): Outputs from the classification head,
              shape (num_decoder_layers, bs, num_queries, cls_out_channels).
              Note cls_out_channels should include background.
            - layers_bbox_preds (Tensor): Sigmoid outputs from the regression
              head with normalized coordinate format (cx, cy, w, h), has shape
              (num_decoder_layers, bs, num_queries, 4).
        """
        layers_cls_scores = self.fc_cls(hidden_states)
        references_before_sigmoid = inverse_sigmoid(references, eps=1e-3)
        tmp_reg_preds = self.fc_reg(hidden_states)
        tmp_reg_preds[..., :references_before_sigmoid.
                      size(-1)] += references_before_sigmoid
        layers_bbox_preds = tmp_reg_preds.sigmoid()
        return layers_cls_scores, layers_bbox_preds

    def predict(self,
                hidden_states: Tensor,
                references: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network. Over-write
        because img_metas are needed as inputs for bbox_head.

        Args:
            hidden_states (Tensor): Feature from the transformer decoder, has
                shape (num_decoder_layers, bs, num_queries, dim).
            references (Tensor): references from the transformer decoder, has
                shape (num_decoder_layers, bs, num_queries, 2/4).
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

        last_layer_hidden_state = hidden_states[-1].unsqueeze(0)
        last_layer_reference = references[-1].unsqueeze(0)
        outs = self(last_layer_hidden_state, last_layer_reference)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions
