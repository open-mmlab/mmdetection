# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_的变体. 目标为高斯分布的浮点值.

    Args:
        pred (torch.Tensor): 网络输出值.
        gaussian_target (torch.Tensor): 网络输出值的拟合目标(高斯分布).
        alpha (float, optional): Focal Loss 中的平衡难易样本参数.默认:2.0.
        gamma (float, optional): 调节负样本loss权重的gamma参数. 默认:4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)  # 正样本权重,仅在gt box出现的地方为1
    neg_weights = (1 - gaussian_target).pow(gamma)  # 负(非正)样本权重,额外添加了一个指数参数
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss


@LOSSES.register_module()
class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss是Focal Loss的一种变体.

    详情参考`paper<https://arxiv.org/abs/1808.01244>`_
    代码修改自 `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    注意,GaussianFocalLoss 中的label target是高斯分布的热力图,而非 0/1 int型值.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): 可选项: "none", "mean" and "sum".
        loss_weight (float): 当前loss的权重.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """前向传播函数.

        Args:
            pred (torch.Tensor): 网络输出值.
            target (torch.Tensor): 网络输出值的拟合目标(高斯分布).
            weight (torch.Tensor, optional): 每个输出值的loss权重. 默认: None.
            avg_factor (int, optional): 用于平均loss的平均因子(一般为正样本个数). 默认: None.
            reduction_override (str, optional): 用于覆盖Loss类初始化中的self.reduction.
                默认为None,表示不覆盖.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_reg = self.loss_weight * gaussian_focal_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg
