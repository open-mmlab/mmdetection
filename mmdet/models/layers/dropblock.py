# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS

eps = 1e-6


@MODELS.register_module()
class DropBlock(nn.Module):
    """Randomly drop some regions of feature maps.

     Please refer to the method proposed in `DropBlock
     <https://arxiv.org/abs/1810.12890>`_ for details.

    Args:
        drop_prob (float): The probability of dropping each block.
        block_size (int): The size of dropped blocks.
        warmup_iters (int): The drop probability will linearly increase
            from `0` to `drop_prob` during the first `warmup_iters` iterations.
            Default: 2000.
    """

    def __init__(self, drop_prob, block_size, warmup_iters=2000, **kwargs):
        super(DropBlock, self).__init__()
        assert block_size % 2 == 1
        assert 0 < drop_prob <= 1
        assert warmup_iters >= 0
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.warmup_iters = warmup_iters
        self.iter_cnt = 0

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map on which some areas will be randomly
                dropped.

        Returns:
            Tensor: The tensor after DropBlock layer.
        """
        if not self.training:
            return x
        self.iter_cnt += 1
        N, C, H, W = list(x.shape)
        gamma = self._compute_gamma((H, W))
        mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))

        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        mask = F.max_pool2d(
            input=mask,
            stride=(1, 1),
            kernel_size=(self.block_size, self.block_size),
            padding=self.block_size // 2)
        mask = 1 - mask
        x = x * mask * mask.numel() / (eps + mask.sum())
        return x

    def _compute_gamma(self, feat_size):
        """Compute the value of gamma according to paper. gamma is the
        parameter of bernoulli distribution, which controls the number of
        features to drop.

        gamma = (drop_prob * fm_area) / (drop_area * keep_area)

        Args:
            feat_size (tuple[int, int]): The height and width of feature map.

        Returns:
            float: The value of gamma.
        """
        gamma = (self.drop_prob * feat_size[0] * feat_size[1])
        gamma /= ((feat_size[0] - self.block_size + 1) *
                  (feat_size[1] - self.block_size + 1))
        gamma /= (self.block_size**2)
        factor = (1.0 if self.iter_cnt > self.warmup_iters else self.iter_cnt /
                  self.warmup_iters)
        return gamma * factor

    def extra_repr(self):
        return (f'drop_prob={self.drop_prob}, block_size={self.block_size}, '
                f'warmup_iters={self.warmup_iters}')
