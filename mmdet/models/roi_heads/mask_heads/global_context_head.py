# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.models.layers import ResLayer, SimplifiedBasicBlock
from mmdet.registry import MODELS
from mmdet.utils import MultiConfig, OptConfigType


@MODELS.register_module()
class GlobalContextHead(BaseModule):
    """Global context head used in `SCNet <https://arxiv.org/abs/2012.10150>`_.

    Args:
        num_convs (int, optional): number of convolutional layer in GlbCtxHead.
            Defaults to 4.
        in_channels (int, optional): number of input channels. Defaults to 256.
        conv_out_channels (int, optional): number of output channels before
            classification layer. Defaults to 256.
        num_classes (int, optional): number of classes. Defaults to 80.
        loss_weight (float, optional): global context loss weight.
            Defaults to 1.
        conv_cfg (dict, optional): config to init conv layer. Defaults to None.
        norm_cfg (dict, optional): config to init norm layer. Defaults to None.
        conv_to_res (bool, optional): if True, 2 convs will be grouped into
            1 `SimplifiedBasicBlock` using a skip connection.
            Defaults to False.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict. Defaults to
            dict(type='Normal', std=0.01, override=dict(name='fc')).
    """

    def __init__(
        self,
        num_convs: int = 4,
        in_channels: int = 256,
        conv_out_channels: int = 256,
        num_classes: int = 80,
        loss_weight: float = 1.0,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        conv_to_res: bool = False,
        init_cfg: MultiConfig = dict(
            type='Normal', std=0.01, override=dict(name='fc'))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.conv_to_res = conv_to_res
        self.fp16_enabled = False

        if self.conv_to_res:
            num_res_blocks = num_convs // 2
            self.convs = ResLayer(
                SimplifiedBasicBlock,
                in_channels,
                self.conv_out_channels,
                num_res_blocks,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
            self.num_convs = num_res_blocks
        else:
            self.convs = nn.ModuleList()
            for i in range(self.num_convs):
                in_channels = self.in_channels if i == 0 else conv_out_channels
                self.convs.append(
                    ConvModule(
                        in_channels,
                        conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(conv_out_channels, num_classes)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function.

        Args:
            feats (Tuple[Tensor]): Multi-scale feature maps.

        Returns:
            Tuple[Tensor]:

                - mc_pred (Tensor): Multi-class prediction.
                - x (Tensor): Global context feature.
        """
        x = feats[-1]
        for i in range(self.num_convs):
            x = self.convs[i](x)
        x = self.pool(x)

        # multi-class prediction
        mc_pred = x.reshape(x.size(0), -1)
        mc_pred = self.fc(mc_pred)

        return mc_pred, x

    def loss(self, pred: Tensor, labels: List[Tensor]) -> Tensor:
        """Loss function.

        Args:
            pred (Tensor): Logits.
            labels (list[Tensor]): Grouth truths.

        Returns:
            Tensor: Loss.
        """
        labels = [lbl.unique() for lbl in labels]
        targets = pred.new_zeros(pred.size())
        for i, label in enumerate(labels):
            targets[i, label] = 1.0
        loss = self.loss_weight * self.criterion(pred, targets)
        return loss
