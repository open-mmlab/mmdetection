# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmdet.registry import MODELS
import torch.nn.functional as F
import numpy as np
from mmengine.dist.utils import is_distributed, get_rank, get_world_size
from mmengine.dist import all_gather
# import diffdist.functional as diff_dist



def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    # out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(get_world_size())]
    out_list = all_gather(x)
    return torch.cat(out_list, dim=0).contiguous()

@MODELS.register_module()
class ContrastiveLoss(nn.Module):

    def __init__(self,
                 loss_weight=0.5,
                 contrast_temperature=0.07,
                 eps=1e-3):
        """Compute dice loss.
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
        """

        super(ContrastiveLoss, self).__init__()
        self.loss_weight = loss_weight
        self.contrast_temperature = contrast_temperature
        self.eps = eps
        self.cross_entropy = nn.CrossEntropyLoss()
        if self.contrast_temperature is not None:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / contrast_temperature))


    def forward(self,
                text_x,
                query):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        image_x = query.float()
        batch_size = image_x.shape[0]
        # get label globally
        if is_distributed():
            labels = torch.arange(batch_size, dtype=torch.long, device=image_x.device) + batch_size * get_rank()
        else:
            labels = torch.arange(batch_size, dtype=torch.long, device=image_x.device)

        # [B, C]
        image_x = F.normalize(image_x.flatten(1), dim=-1)
        text_x = F.normalize(text_x.flatten(1), dim=-1)

        if is_distributed():
            logits_per_img = image_x @ dist_collect(text_x).t()
            logits_per_text = text_x @ dist_collect(image_x).t()
        else:
            logits_per_img = image_x @ text_x.t()
            logits_per_text = text_x @ image_x.t()

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = self.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = self.cross_entropy(logits_per_text * logit_scale, labels)

        loss_contrastive = loss_img + loss_text

        loss = self.loss_weight * loss_contrastive
        return loss