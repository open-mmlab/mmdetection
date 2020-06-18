# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.distributed as dist
from mmcv.cnn import NORM_LAYERS
from mmcv.runner import get_dist_info
from torch import nn
from torch.autograd.function import Function


class AllReduce(Function):

    @staticmethod
    def forward(ctx, input):
        input_list = [
            torch.zeros_like(input) for k in range(dist.get_world_size())
        ]
        # Use allgather instead of allreduce since I don't trust in-place
        # operations .
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


@NORM_LAYERS.register_module('NaiveSyncBN')
class NaiveSyncBatchNorm(nn.BatchNorm2d):
    """
    `torch.nn.SyncBatchNorm` has known unknown bugs.
    It produces significantly worse AP (and sometimes goes NaN)
    when the batch size on each worker is quite different
    (e.g., when scale augmentation is used, or when it is applied to mask
    head).

    Use this implementation before `nn.SyncBatchNorm` is fixed.
    It is slower than `nn.SyncBatchNorm`.

    Note:
        There isn't a single definition of Sync BatchNorm.

        When ``stats_mode==""``, this module computes overall statistics by
        using statistics of each worker with equal weight. The result is
        true statistics of all samples (as if they are all on one worker)
        only when all workers have the same (N, H, W). This mode does not
        support inputs with zero batch size.

        When ``stats_mode=="N"``, this module computes overall statistics by
        weighting the statistics of each worker by their ``N``. The result
        is true statistics of all samples (as if they are all on one worker)
        only when all workers have the same (H, W).
        It is slower than ``stats_mode==""``.

        Even though the result of this module may not be the true statistics
        of all samples, it may still be reasonable because it might be
        preferrable to assign equal weights to all workers, regardless of
        their (H, W) dimension, instead of putting larger weight on larger
        images. From preliminary experiments, little difference is found
        between such a simplified implementation and an accurate computation
        of overall mean & variance.
    """

    def __init__(self, *args, stats_mode='', **kwargs):
        super().__init__(*args, **kwargs)
        assert stats_mode in ['', 'N']
        self._stats_mode = stats_mode

    def forward(self, input):
        _, world_size = get_dist_info()
        if world_size == 1 or not self.training:
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]

        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        if self._stats_mode == '':
            assert B > 0, 'SyncBatchNorm(stats_mode="") does not support ' \
                          'zero batch size.'
            vec = torch.cat([mean, meansqr], dim=0)
            vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())
            mean, meansqr = torch.split(vec, C)
            momentum = self.momentum
        else:
            if B == 0:
                vec = torch.zeros([2 * C + 1],
                                  device=mean.device,
                                  dtype=mean.dtype)
                # make sure there is gradient w.r.t input
                vec = vec + input.sum()
            else:
                vec = torch.cat([
                    mean, meansqr,
                    torch.ones([1], device=mean.device, dtype=mean.dtype)
                ],
                                dim=0)
            vec = AllReduce.apply(vec * B)

            total_batch = vec[-1].detach()
            momentum = total_batch.clamp(
                max=1) * self.momentum  # no update if total_batch is 0
            total_batch = torch.max(
                total_batch, torch.ones_like(total_batch))  # avoid div-by-zero
            mean, meansqr, _ = torch.split(vec / total_batch, C)

        var = meansqr - mean * mean
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)

        self.running_mean += momentum * (mean.detach() - self.running_mean)
        self.running_var += momentum * (var.detach() - self.running_var)
        return input * scale + bias
