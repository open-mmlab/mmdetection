# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional

import torch
import torch.nn as nn
from mmengine.model import ExponentialMovingAverage
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class ExpMomentumEMA(ExponentialMovingAverage):
    """Exponential moving average (EMA) with exponential momentum strategy,
    which is used in YOLOX.

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The momentum used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `averaged_param = (1-momentum) * averaged_param + momentum *
           source_param`. Defaults to 0.0002.
        gamma (int): Use a larger momentum early in training and gradually
            annealing to a smaller value to update the ema model smoothly. The
            momentum is calculated as
            `(1 - momentum) * exp(-(1 + steps) / gamma) + momentum`.
            Defaults to 2000.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(self,
                 model: nn.Module,
                 momentum: float = 0.0002,
                 gamma: int = 2000,
                 interval=1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        super().__init__(
            model=model,
            momentum=momentum,
            interval=interval,
            device=device,
            update_buffers=update_buffers)
        assert gamma > 0, f'gamma must be greater than 0, but got {gamma}'
        self.gamma = gamma

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the moving average of the parameters using the exponential
        momentum strategy.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        momentum = (1 - self.momentum) * math.exp(
            -float(1 + steps) / self.gamma) + self.momentum
        averaged_param.mul_(1 - momentum).add_(source_param, alpha=momentum)
