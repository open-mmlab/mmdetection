import torch
import torch.nn as nn


class ToFP16(nn.Module):

    def __init__(self):
        super(ToFP16, self).__init__()

    def forward(self, x):
        return x.half()


# copy updated param from fp32_weight to fp16 net
def copy_in_params(fp16_net, fp32_weight):
    for net_param, fp32_weight_param in zip(fp16_net.parameters(),
                                            fp32_weight):
        net_param.data.copy_(fp32_weight_param.data)


# copy gradient from fp16 net to fp32 weight copy
def set_grad(fp16_net, fp32_weight):
    for param, param_w_grad in zip(fp32_weight, fp16_net.parameters()):
        if param_w_grad.grad is not None:
            if param.grad is None:
                param.grad = param.data.new(*param.data.size())
            param.grad.data.copy_(param_w_grad.grad.data)


# convert batch norm layer to fp32
def bn_convert_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        bn_convert_float(child)
    return module


class WrappedBN(nn.BatchNorm2d):

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 fp16=False):
        super(WrappedBN, self).__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats)
        self.fp16 = fp16

    def forward(self, input):
        if self.fp16:
            return super(WrappedBN, self).forward(input.float()).half()
        else:
            return super(WrappedBN, self).forward(input)


class WrappedGN(nn.GroupNorm):

    def __init__(self,
                 num_groups,
                 num_channels,
                 eps=1e-5,
                 affine=True,
                 fp16=False):
        super(WrappedGN, self).__init__(
            num_groups, num_channels, eps=eps, affine=affine)
        self.fp16 = fp16

    def forward(self, input):
        if self.fp16:
            return super(WrappedGN, self).forward(input.float()).half()
        else:
            return super(WrappedGN, self).forward(input)
