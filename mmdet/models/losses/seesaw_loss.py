import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .accuracy import accuracy


def _multi_label_cross_entropy_loss(cls_score, m_labels, m_label_weights,
                                    avg_factor):
    pos_inds = m_labels * m_label_weights
    neg_inds = (1 - m_labels) * m_label_weights
    pos_inds_count = pos_inds.sum(dim=1)
    neg_inds_count = neg_inds.sum(dim=1)
    unique_pos_count = pos_inds_count.unique()
    unique_neg_count = neg_inds_count.unique()
    losses = []
    for u_p in unique_pos_count:
        for u_n in unique_neg_count:
            slice_inds = (pos_inds_count == u_p) & (neg_inds_count == u_n)
            if slice_inds.sum() == 0:
                continue
            slice_cls_score = cls_score[slice_inds, :]
            slice_pos_inds = pos_inds[slice_inds, :]
            slice_neg_inds = neg_inds[slice_inds, :]
            sample_num = slice_cls_score.size(0)
            slice_pos_score = slice_cls_score[slice_pos_inds == 1].view(
                sample_num, -1)
            slice_neg_score = slice_cls_score[slice_neg_inds == 1].view(
                sample_num, -1)
            assert slice_pos_score.size(1) == u_p
            assert slice_neg_score.size(1) == u_n
            pos_logsumexp = torch.logsumexp(-slice_pos_score, 1)
            neg_logsumexp = torch.logsumexp(slice_neg_score, 1)
            inv_pos_num_log = torch.log(1 / slice_pos_inds.sum(dim=1))
            loss = F.softplus(inv_pos_num_log + pos_logsumexp +
                              neg_logsumexp).sum()
            losses.append(loss)
    return sum(losses) / avg_factor


def _seesaw_ce_loss(cls_score, m_labels, m_label_weights, labels, cum_samples,
                    p, q, eps, avg_factor):
    # only need pos samples

    assert m_labels[:, -1].sum() == 0
    seesaw_weights = cls_score.new_ones(m_labels.size())
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp_min(
            1) / cum_samples[:, None].clamp_min(1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor

    if q > 0:
        scores = F.softmax(cls_score, dim=1).detach()
        self_scores = scores[torch.arange(0, len(scores)).cuda().long(),
                             labels.long()]
        score_matrix = scores / self_scores[:, None].clamp_min(eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights[:, :-1] = seesaw_weights[:, :-1] * compensation_factor

    cls_score = cls_score + (seesaw_weights.log() * (1 - m_labels))[:, :-1]

    return _multi_label_cross_entropy_loss(cls_score, m_labels[:, :-1],
                                           m_label_weights[:, :-1], avg_factor)


@LOSSES.register_module()
class SeesawLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 p=0.8,
                 q=2.0,
                 num_classes=1203,
                 eps=1e-2,
                 loss_weight=1.0,
                 return_dict=True):
        super(SeesawLoss, self).__init__()
        assert not use_sigmoid
        self.use_sigmoid = False
        self.p = p
        self.q = q
        self.loss_weight = loss_weight
        self.num_classes = num_classes

        self.eps = eps

        # 0 for pos, 1 for neg
        self.cls_criterion = _seesaw_ce_loss

        self.register_buffer(
            'cum_samples',
            torch.Tensor(self.num_classes + 1).float().cuda().fill_(0))

        self.custom_cls_channels = True
        self.custom_acc = True
        self.return_dict = return_dict

    def _expand_binary_labels(self, labels, label_weights, label_channels):
        bin_labels = labels.new_full((labels.size(0), label_channels), 0)
        inds = torch.arange(0, labels.size(0)).cuda().long()
        bin_labels[inds, labels[inds]] = 1
        if label_weights is None:
            bin_label_weights = None
        else:
            bin_label_weights = label_weights.view(-1, 1).expand(
                label_weights.size(0), label_channels)
        return bin_labels, bin_label_weights

    def get_cls_channels(self, num_classes):
        # num_classes + pos + neg
        assert num_classes == self.num_classes
        return num_classes + 2

    def _split_cls_score(self, cls_score):
        cls_score_classes = cls_score[..., :-2]
        cls_score_binary = cls_score[..., -2:]
        return cls_score_classes, cls_score_binary

    def get_activation(self, cls_score):
        cls_score_classes, cls_score_binary = self._split_cls_score(cls_score)
        cls_score_classes = F.softmax(cls_score_classes, dim=-1)
        cls_score_binary = F.softmax(cls_score_binary, dim=-1)
        cls_score_pos = cls_score_binary[..., [0]]
        cls_score_neg = cls_score_binary[..., [1]]
        cls_score_classes = cls_score_classes * cls_score_pos
        scores = torch.cat([cls_score_classes, cls_score_neg], dim=-1)
        return scores

    def get_accuracy(self, cls_score, labels):
        pos_inds = labels < self.num_classes
        b_labels = (labels == self.num_classes).long()
        cls_score_classes, cls_score_binary = self._split_cls_score(cls_score)
        acc_binary = accuracy(cls_score_binary, b_labels)
        acc_classes = accuracy(cls_score_classes[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_binary'] = acc_binary
        acc['acc_classes'] = acc_classes
        return acc

    def forward(self,
                cls_score,
                labels,
                label_weights,
                avg_factor,
                reduction_override=None):
        m_labels, m_label_weights = self._expand_binary_labels(
            labels, label_weights, self.num_classes + 1)
        assert (m_labels[torch.arange(0, labels.size(0)).cuda().long(),
                         labels] == 0).sum() == 0
        pos_inds = labels < self.num_classes
        # 0 for pos, 1 for neg
        b_labels = (labels == self.num_classes).long()
        unique_labels = labels.unique()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            self.cum_samples[u_l] += inds_.sum()

        cls_score_classes, cls_score_binary = self._split_cls_score(cls_score)

        loss_cls_classes = self.loss_weight * self.cls_criterion(
            cls_score_classes[pos_inds], m_labels[pos_inds, :],
            m_label_weights[pos_inds, :], labels[pos_inds], self.cum_samples,
            self.p, self.q, self.eps, avg_factor)

        loss_cls_binary = F.cross_entropy(
            cls_score_binary, b_labels, weight=None, reduction='none')
        loss_cls_binary = (loss_cls_binary * label_weights).sum() / avg_factor
        if self.return_dict:
            loss_cls = dict()
            loss_cls['loss_cls_binary'] = loss_cls_binary
            loss_cls['loss_cls_classes'] = loss_cls_classes
        else:
            loss_cls = loss_cls_classes + loss_cls_binary
        return loss_cls
