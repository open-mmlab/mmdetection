# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.layers import ResLayer, SimplifiedBasicBlock
from mmdet.registry import MODELS
from .fcn_mask_head import FCNMaskHead


@MODELS.register_module()
class SCNetMaskHead(FCNMaskHead):
    """Mask head for `SCNet <https://arxiv.org/abs/2012.10150>`_.

    Args:
        conv_to_res (bool, optional): if True, change the conv layers to
            ``SimplifiedBasicBlock``.
    """

    def __init__(self, conv_to_res: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv_to_res = conv_to_res
        if conv_to_res:
            assert self.conv_kernel_size == 3
            self.num_res_blocks = self.num_convs // 2
            self.convs = ResLayer(
                SimplifiedBasicBlock,
                self.in_channels,
                self.conv_out_channels,
                self.num_res_blocks,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
