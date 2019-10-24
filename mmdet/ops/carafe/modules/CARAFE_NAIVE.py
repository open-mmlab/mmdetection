from torch.nn.modules.module import Module

from ..functions.carafe_naive import CARAFENAIVEFunction


class CARAFENAIVE(Module):

    def __init__(self, kernel_size, group_size, scale_factor):
        super(CARAFENAIVE, self).__init__()

        self.kernel_size = int(kernel_size)
        self.group_size = int(group_size)
        self.scale_factor = int(scale_factor)

    def forward(self, features, masks):
        return CARAFENAIVEFunction.apply(features, masks, self.kernel_size,
                                         self.group_size, self.scale_factor)
