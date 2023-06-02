# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

from torch import Tensor

from mmdet.registry import MODELS
from .convfc_bbox_head import ConvFCBBoxHead


@MODELS.register_module()
class SCNetBBoxHead(ConvFCBBoxHead):
    """BBox head for `SCNet <https://arxiv.org/abs/2012.10150>`_.

    This inherits ``ConvFCBBoxHead`` with modified forward() function, allow us
    to get intermediate shared feature.
    """

    def _forward_shared(self, x: Tensor) -> Tensor:
        """Forward function for shared part.

        Args:
            x (Tensor): Input feature.

        Returns:
            Tensor: Shared feature.
        """
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        return x

    def _forward_cls_reg(self, x: Tensor) -> Tuple[Tensor]:
        """Forward function for classification and regression parts.

        Args:
            x (Tensor): Input feature.

        Returns:
            tuple[Tensor]:

                - cls_score (Tensor): classification prediction.
                - bbox_pred (Tensor): bbox prediction.
        """
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score, bbox_pred

    def forward(
            self,
            x: Tensor,
            return_shared_feat: bool = False) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.

        Args:
            x (Tensor): input features
            return_shared_feat (bool): If True, return cls-reg-shared feature.

        Return:
            out (tuple[Tensor]): contain ``cls_score`` and ``bbox_pred``,
                if  ``return_shared_feat`` is True, append ``x_shared`` to the
                returned tuple.
        """
        x_shared = self._forward_shared(x)
        out = self._forward_cls_reg(x_shared)

        if return_shared_feat:
            out += (x_shared, )

        return out
