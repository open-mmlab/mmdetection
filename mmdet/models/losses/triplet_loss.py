# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet.registry import MODELS


@MODELS.register_module()
class TripletLoss(BaseModule):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for
            Person Re-Identification. arXiv:1703.07737.
    Imported from `<https://github.com/KaiyangZhou/deep-person-reid/blob/
        master/torchreid/losses/hard_mine_triplet_loss.py>`_.
    Args:
        margin (float, optional): Margin for triplet loss. Defaults to 0.3.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        hard_mining (bool, optional): Whether to perform hard mining.
            Defaults to True.
    """

    def __init__(self,
                 margin: float = 0.3,
                 loss_weight: float = 1.0,
                 hard_mining=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.loss_weight = loss_weight
        self.hard_mining = hard_mining

    def hard_mining_triplet_loss_forward(
            self, inputs: torch.Tensor,
            targets: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape
                (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape
                (num_classes).

        Returns:
            torch.Tensor: triplet loss with hard mining.
        """

        batch_size = inputs.size(0)

        # Compute Euclidean distance
        dist = torch.pow(inputs, 2).sum(
            dim=1, keepdim=True).expand(batch_size, batch_size)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the furthest positive sample
        # and nearest negative sample in the embedding space
        mask = targets.expand(batch_size, batch_size).eq(
            targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.loss_weight * self.ranking_loss(dist_an, dist_ap, y)

    def forward(self, inputs: torch.Tensor,
                targets: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape
                (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape
                (num_classes).

        Returns:
            torch.Tensor: triplet loss.
        """
        if self.hard_mining:
            return self.hard_mining_triplet_loss_forward(inputs, targets)
        else:
            raise NotImplementedError()
