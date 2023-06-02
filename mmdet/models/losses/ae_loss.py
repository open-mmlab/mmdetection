# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


def ae_loss_per_image(tl_preds, br_preds, match):
    """单张图片上的Associative Embedding Loss.

    Associative Embedding Loss 包含两个部分: pull loss 和 push loss.
    Pull loss 使来自同一gt box的embedding vectors彼此更接近.
    Push loss 区分不同gt box之间的embedding vector,使得它们之间的差距足够大.

    在计算时,通常有以下三种情况:
        - 图片中没有gt box: pull loss 和 push loss 都将为0.
        - 图片中有1个gt box: push loss将为0, pull loss将由gt box的两个corner来计算.
        - 图像中有大于1个gt box: pull loss由每个gt box的一对corner计算, push loss由
            每个gt box与所有其他gt box计算.我们使用对角线上值为0的混淆矩阵来计算pull loss.

    Args:
        tl_preds (tensor): 左上corner的 Embedding特征图. [corner_emb_channels, h, w]
        br_preds (tensor): 右下corner的 Embedding特征图. [corner_emb_channels, h, w]
        match (list): gt box在下采样4倍后的特征图上所属特征点的坐标.
            [[[y1, x1], [y2, x2]],] * num_gt
    """

    tl_list, br_list, me_list = [], [], []
    if len(match) == 0:  # 图片中没有gt box
        pull_loss = tl_preds.sum() * 0.
        push_loss = tl_preds.sum() * 0.
    else:
        for m in match:
            [tl_y, tl_x], [br_y, br_x] = m
            tl_e = tl_preds[:, tl_y, tl_x].view(-1, 1)  # [corner_emb_channels, 1]
            br_e = br_preds[:, br_y, br_x].view(-1, 1)
            tl_list.append(tl_e)
            br_list.append(br_e)
            me_list.append((tl_e + br_e) / 2.0)

        tl_list = torch.cat(tl_list)  # [num_gt*corner_emb_channels, 1]
        br_list = torch.cat(br_list)
        me_list = torch.cat(me_list)

        assert tl_list.size() == br_list.size()

        # N为num_gt*corner_emb_channels, M恒为1(由于上面for循环中view第二参数指定为1).
        # 所以下面所有涉及N的变量都不太准确.不过由于corner_emb_channels默认为1所以影响不大.
        N, M = tl_list.size()

        pull_loss = (tl_list - me_list).pow(2) + (br_list - me_list).pow(2)
        pull_loss = pull_loss.sum() / N

        margin = 1  # CornerNet的tl_emb与br_emb的绝对差值上限,详见论文3.3节

        # push loss的混淆矩阵
        # 这里假设corner_emb_channels=1,假设me_list为
        # tensor([[-0.2413],
        #         [-0.4207]], device='cuda:0', grad_fn=<CatBackward0>)
        # 上述tensor表明在一张图片上有两个gt,再进行expand操作为
        # tensor([[[-0.2413],
        #          [-0.4207]],
        #
        #         [[-0.2413],
        #          [-0.4207]]], device='cuda:0', grad_fn=<ExpandBackward0>)
        # 再进行permute操作为
        # tensor([[[-0.2413],
        #          [-0.2413]],
        #
        #         [[-0.4207],
        #          [-0.4207]]], device='cuda:0', grad_fn=<PermuteBackward0>)
        # 再减去me_list(需要对广播机制很熟悉),得以构造出每个gt与其余gt的Embedding差距的混淆矩阵
        # 注意这个混淆矩阵是3维的,第三维表示一维的Embedding向量
        # 显然conf_mat的第一索引与第二索引相同的位置为0,因为相同gt之间的Embedding差距自然为0
        # 实际上该操作等于me_list[:,None].repeat(1, N, 1)
        # conf_mat表示不同gt之间的Embedding向量差距
        conf_mat = me_list.expand((N, N, M)).permute(1, 0, 2) - me_list
        # 除了对角线,其余位置全为1作为计算权重.即只计算不同gt之间的Embedding loss
        conf_weight = 1 - torch.eye(N).type_as(me_list)
        # 由于conf_mat的最后一个维度大小固定为1,所以这里sum(-1)等价于squeeze行为, -> [N, N]
        # margin参数的作用充当是不同gt之间的emb绝对差值的标准值,即差值越接近于margin则loss越小
        # 但margin也是一个超参数,论文中在所有实验下都设置为1.
        conf_mat = conf_weight * (margin - conf_mat.sum(-1).abs())

        if N > 1:  # 图像中有大于1个gt box,
            # 因为除了对角线上的N个,其余都计算在内,所以分母为N * (N - 1)
            # 不同gt之间的emb绝对差值可能大于1,在上面的计算过程中会出现负数,所以这里需要限制为0.
            push_loss = F.relu(conf_mat).sum() / (N * (N - 1))
        else:
            push_loss = tl_preds.sum() * 0.

    return pull_loss, push_loss


@MODELS.register_module()
class AssociativeEmbeddingLoss(nn.Module):
    """Associative Embedding Loss.

    详情参考
    `Associative Embedding <https://arxiv.org/abs/1611.05424>`_ 和
    `CornerNet <https://arxiv.org/abs/1808.01244>`_ .
    代码修改自 `kp_utils.py <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L180>`_  # noqa: E501

    Args:
        pull_weight (float): 来自同一gt box的pull loss权重.
        push_weight (float): 来自不同gt box的push loss权重.
    """

    def __init__(self, pull_weight=0.25, push_weight=0.25):
        super(AssociativeEmbeddingLoss, self).__init__()
        self.pull_weight = pull_weight
        self.push_weight = push_weight

    def forward(self, pred, target, match):
        """Forward function."""
        batch = pred.size(0)
        pull_all, push_all = 0.0, 0.0
        for i in range(batch):
            pull, push = ae_loss_per_image(pred[i], target[i], match[i])

            pull_all += self.pull_weight * pull
            push_all += self.push_weight * push

        return pull_all, push_all
