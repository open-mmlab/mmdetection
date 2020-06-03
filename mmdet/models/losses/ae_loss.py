import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def ae_loss_per_image(tl_preds, br_preds, match):
    tl_list, br_list, me_list = [], [], []
    for m in match:
        [tl_y, tl_x], [br_y, br_x] = m
        tl_e = tl_preds[:, tl_y, tl_x].view(-1, 1)
        br_e = br_preds[:, br_y, br_x].view(-1, 1)
        tl_list.append(tl_e)
        br_list.append(br_e)
        me_list.append((tl_e + br_e) / 2.0)

    tl_list = torch.cat(tl_list)
    br_list = torch.cat(br_list)
    me_list = torch.cat(me_list)

    # assert tl_list.size() == br_list.size()

    N, M = tl_list.size()

    if N > 0:
        pull_loss = (tl_list - me_list).pow(2) + (br_list - me_list).pow(2)
        pull_loss = pull_loss.sum() / N
    else:
        pull_loss = torch.tensor(0).type_as(me_list).requires_grad_()

    margin = 1
    push_loss = torch.tensor(0).type_as(me_list).requires_grad_()

    # confusion matrix of push loss
    conf_mat = (me_list.expand((N, N, M)).permute(1, 0, 2) - me_list).sum(-1)
    conf_mat = (1 - torch.eye(N).type_as(me_list)) * (margin - conf_mat.abs())

    if N > 1:
        push_loss = F.relu(conf_mat).sum() / (N * (N - 1))

    return pull_loss, push_loss


@LOSSES.register_module()
class AELoss(nn.Module):
    """ Associative Embedding Loss.

    Please refer to https://arxiv.org/abs/1611.05424 for more details.
    Code is modified from https://github.com/princeton-vl/CornerNet.

    Args:
        pull_weight (float): Loss weight for corners from same object.
        push_weight (float): Loss weight for corners from different object.
    """

    def __init__(self, pull_weight=0.25, push_weight=0.25):
        super(AELoss, self).__init__()
        self.pull_weight = pull_weight
        self.push_weight = push_weight

    def forward(self, pred, target, match):
        batch = pred.size(0)
        pull_all, push_all = 0.0, 0.0
        for i in range(batch):
            pull, push = ae_loss_per_image(pred[i], target[i], match[i])

            pull_all += self.pull_weight * pull
            push_all += self.push_weight * push

        return pull_all, push_all
