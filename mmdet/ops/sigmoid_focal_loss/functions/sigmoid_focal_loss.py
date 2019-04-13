from torch.autograd import Function
from torch.autograd.function import once_differentiable

from .. import sigmoid_focal_loss


class _SigmoidFocalLoss(Function):

    @staticmethod
    def forward(ctx, logits, targets, gamma, alpha):
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha

        losses = sigmoid_focal_loss.forward(logits, targets, num_classes,
                                            gamma, alpha)
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_logits = sigmoid_focal_loss.backward(logits, targets, d_loss,
                                               num_classes, gamma, alpha)
        return d_logits, None, None, None, None


sigmoid_focal_loss_cuda = _SigmoidFocalLoss.apply
