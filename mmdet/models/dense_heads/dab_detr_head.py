# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor

from mmdet.registry import MODELS
from ..layers import MLP, inverse_sigmoid
from ..utils import multi_apply
from .detr_head import DETRHead


@MODELS.register_module()
class DABDETRHead(DETRHead):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        TODO
    """

    _version = 2

    def __init__(
            self,
            *args,
            # bbox_embed_diff_each_layer=False,
            **kwargs) -> None:
        # self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        super(DABDETRHead, self).__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the transformer head."""
        # cls branch
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        # reg branch
        self.fc_reg = MLP(self.embed_dims, self.embed_dims, 4, 3)
        # self.activate = nn.ReLU()
        # self.reg_ffn = FFN(
        #     self.embed_dims,
        #     self.embed_dims,
        #     self.num_reg_fcs,
        #     dict(type='ReLU', inplace=True),
        #     dropout=0.0,
        #     add_residual=False)
        # # NOTE the activations of reg_branch is the same as those in
        # # transformer, but they are actually different in Conditional DETR
        # # and DAB DETR (prelu in transformer and relu in reg_branch)
        # self.fc_reg = Linear(self.embed_dims, 4)

    # def _load_from_state_dict  # TODO

    def init_weights(self) -> None:
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
        constant_init(self.fc_reg.layers[-1], 0., bias=0.)

    def forward_single(self, outs_dec: Tensor,
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

    def forward(self, x: List[Tensor],
                refs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward function.

        Args:
            x:
            TODO

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

            - all_cls_scores_list (list[Tensor]): Classification scores \
            for each scale level. Each is a 4D-tensor with shape \
            [nb_dec, bs, num_query, cls_out_channels]. Note \
            `cls_out_channels` should includes background.
            - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
            outputs for each scale level. Each is a 4D-tensor with \
            normalized coordinate format (cx, cy, w, h) and shape \
            [nb_dec, bs, num_query, 4].
        """

        return multi_apply(self.forward_single, x, refs)
