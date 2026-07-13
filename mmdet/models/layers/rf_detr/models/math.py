# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Extracted from lwdetr.py (Phase 10)
# Original copyrights: LW-DETR (Baidu), Conditional DETR (Microsoft),
# DETR (Facebook), Deformable DETR (SenseTime)
# ------------------------------------------------------------------------
"""Mathematical building blocks: MLP, inverse_sigmoid, accuracy, interpolate."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    """Computes the precision@k for the specified values of k."""
    # Context manager avoids mypy untyped-decorator on @torch.no_grad()
    # Revert to decorator once torch floor >= 2.10 (pytorch/pytorch#166413)
    with torch.no_grad():
        if target.numel() == 0:
            return [torch.zeros([], device=output.device)]
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def interpolate(
    input: Tensor,
    size: list[int] | None = None,
    scale_factor: float | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
) -> Tensor:
    """Equivalent to nn.functional.interpolate.

    Historically this wrapped ``torchvision`` to support empty batch sizes on old releases; modern ``torch`` (>=2.2)
    handles empty batches natively, so this now delegates directly to :func:`torch.nn.functional.interpolate`.
    """
    interpolated: Tensor = F.interpolate(input, size, scale_factor, mode, align_corners)
    return interpolated


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute element-wise logit (inverse sigmoid) of a tensor.

    Clamps input to ``[0, 1]``, then returns ``log(x / (1 - x))`` with
    numerical stability guarded by ``eps``.

    Args:
        x: Input tensor with values in the range ``[0, 1]``.
        eps: Minimum clamp value applied to ``x`` and ``1 - x`` before the
            log operation. Guards against ``log(0)``.

    Returns:
        Tensor of the same shape as ``x`` containing the logit values.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the MLP: ReLU on all layers except the last.

        Args:
            x: Input tensor of shape ``(..., input_dim)``.

        Returns:
            Output tensor of shape ``(..., output_dim)``.
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
