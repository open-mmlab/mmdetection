from torch import nn
from ..functions.psroi_pool import psroi_pool


class PSRoIPool(nn.Module):

    def __init__(self, out_size, spatial_scale, group_size):
        super(PSRoIPool, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)

    def forward(self, features, rois):
        return psroi_pool(features, rois, self.out_size, self.spatial_scale,
                          self.group_size)
