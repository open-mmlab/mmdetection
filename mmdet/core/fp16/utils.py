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


class WrapedBN(nn.BatchNorm2d):

    def forward(self, input):
        return super(WrapedBN, self).forward(input.float()).half()


class WrapedGN(nn.GroupNorm):

    def forward(self, input):
        return super(WrapedGN, self).forward(input.float()).half()
