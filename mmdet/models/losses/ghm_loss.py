import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


@LOSSES.register_module
class GHMC(nn.Module):
    def __init__(
            self,
            bins=10,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """ Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary class target for each sample.
        label_weight [batch_num, class_num]:
            the value is 1 if the sample is valid and 0 if ignored.
        """
        if not self.use_sigmoid:
            raise NotImplementedError
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_binary_labels(
                                    target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight


@LOSSES.register_module
class GHMR(nn.Module):
    def __init__(
            self,
            mu=0.02,
            bins=10,
            momentum=0,
            loss_weight=1.0):
        super(GHMR, self).__init__()
        self.mu = mu
        self.bins = bins
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, avg_factor=None):
        """ Args:
        pred [batch_num, 4 (* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num, 4 (* class_num)]:
            The target regression values with the same size of pred.
        label_weight [batch_num, 4 (* class_num)]:
            The weight of each sample, 0 if ignored.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = pred - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu

        # gradient length
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        weights = torch.zeros_like(g)

        valid = label_weight > 0
        tot = max(label_weight.float().sum().item(), 1.0)
        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n

        loss = loss * weights
        loss = loss.sum() / tot
        return loss * self.loss_weight
