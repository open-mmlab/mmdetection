# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

from mmcv.cnn import ConvModule
from torch import Tensor

from mmdet.registry import MODELS
from .fcn_mask_head import FCNMaskHead


@MODELS.register_module()
class HTCMaskHead(FCNMaskHead):
    """Mask head for HTC.

    Args:
        with_conv_res (bool): Whether add conv layer for ``res_feat``.
            Defaults to True.
    """

    def __init__(self, with_conv_res: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.with_conv_res = with_conv_res
        if self.with_conv_res:
            self.conv_res = ConvModule(
                self.conv_out_channels,
                self.conv_out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)

    def forward(self,
                x: Tensor,
                res_feat: Optional[Tensor] = None,
                return_logits: bool = True,
                return_feat: bool = True) -> Union[Tensor, List[Tensor]]:
        """
        Args:
            x (Tensor): Feature map.
            res_feat (Tensor, optional): Feature for residual connection.
                Defaults to None.
            return_logits (bool): Whether return mask logits. Defaults to True.
            return_feat (bool): Whether return feature map. Defaults to True.

        Returns:
            Union[Tensor, List[Tensor]]: The return result is one of three
                results: res_feat, logits, or [logits, res_feat].
        """
        assert not (not return_logits and not return_feat)
        if res_feat is not None:
            assert self.with_conv_res
            res_feat = self.conv_res(res_feat)
            x = x + res_feat
        for conv in self.convs:
            x = conv(x)
        res_feat = x
        outs = []
        if return_logits:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
            mask_preds = self.conv_logits(x)
            outs.append(mask_preds)
        if return_feat:
            outs.append(res_feat)
        return outs if len(outs) > 1 else outs[0]
