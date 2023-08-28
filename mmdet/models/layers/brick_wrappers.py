# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.wrappers import NewEmptyTensorOp, obsolete_torch_version

from mmdet.registry import MODELS

if torch.__version__ == 'parrots':
    TORCH_VERSION = torch.__version__
else:
    # torch.__version__ could be 1.3.1+cu92, we only need the first two
    # for comparison
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])


def adaptive_avg_pool2d(input, output_size):
    """Handle empty batch dimension to adaptive_avg_pool2d.

    Args:
        input (tensor): 4D tensor.
        output_size (int, tuple[int,int]): the target output size.
    """
    if input.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 9)):
        if isinstance(output_size, int):
            output_size = [output_size, output_size]
        output_size = [*input.shape[:2], *output_size]
        empty = NewEmptyTensorOp.apply(input, output_size)
        return empty
    else:
        return F.adaptive_avg_pool2d(input, output_size)


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """Handle empty batch dimension to AdaptiveAvgPool2d."""

    def forward(self, x):
        # PyTorch 1.9 does not support empty tensor inference yet
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 9)):
            output_size = self.output_size
            if isinstance(output_size, int):
                output_size = [output_size, output_size]
            else:
                output_size = [
                    v if v is not None else d
                    for v, d in zip(output_size,
                                    x.size()[-2:])
                ]
            output_size = [*x.shape[:2], *output_size]
            empty = NewEmptyTensorOp.apply(x, output_size)
            return empty

        return super().forward(x)


# Modified from
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py#L13 # noqa
@MODELS.register_module('FrozenBN')
class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are
    fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.
    Args:
       num_features (int):  :math:`C` from an expected input of size
            :math:`(N, C, H, W)`.
       eps (float): a value added to the denominator for numerical stability.
            Default: 1e-5
    """

    def __init__(self, num_features, eps=1e-5, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias
            # as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def __repr__(self):
        return 'FrozenBatchNorm2d(num_features={}, eps={})'.format(
            self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):
        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.
        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res
