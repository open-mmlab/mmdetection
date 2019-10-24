from torch.nn.modules.module import Module

from ..functions.carafe import CARAFEFunction


class CARAFE(Module):

    def __init__(self, kernel_size, group_size, scale_factor, benchmark=False):
        super(CARAFE, self).__init__()

        self.kernel_size = int(kernel_size)
        self.group_size = int(group_size)
        self.scale_factor = int(scale_factor)
        self.benchmark = benchmark

    def forward(self, features, masks):
        return CARAFEFunction.apply(features, masks, self.kernel_size,
                                    self.group_size, self.scale_factor,
                                    self.benchmark)
