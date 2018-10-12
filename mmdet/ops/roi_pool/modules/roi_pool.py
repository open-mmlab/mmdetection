from torch.nn.modules.module import Module
from ..functions.roi_pool import roi_pool


class RoIPool(Module):

    def __init__(self, out_size, spatial_scale):
        super(RoIPool, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return roi_pool(features, rois, self.out_size, self.spatial_scale)
