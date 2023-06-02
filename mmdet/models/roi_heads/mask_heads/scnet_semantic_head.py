# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import HEADS
from mmdet.models.utils import ResLayer, SimplifiedBasicBlock
from .fused_semantic_head import FusedSemanticHead


@HEADS.register_module()
class SCNetSemanticHead(FusedSemanticHead):
    """Mask head for `SCNet <https://arxiv.org/abs/2012.10150>`_.

    Args:
        conv_to_res (bool, optional): if True, change the conv layers to
            ``SimplifiedBasicBlock``.
    """

    def __init__(self, conv_to_res=True, **kwargs):
        super(SCNetSemanticHead, self).__init__(**kwargs)
        self.conv_to_res = conv_to_res
        if self.conv_to_res:
            num_res_blocks = self.num_convs // 2
            self.convs = ResLayer(
                SimplifiedBasicBlock,
                self.in_channels,
                self.conv_out_channels,
                num_res_blocks,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
            self.num_convs = num_res_blocks
