import torch.nn as nn
from torch.nn.modules.utils import _pair

from ..functions.roi_pool import roi_pool


class RoIPool(nn.Module):

    def __init__(self, out_size, spatial_scale, use_torchvision=False):
        super(RoIPool, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.use_torchvision = use_torchvision

    def forward(self, features, rois):
        if self.use_torchvision:
            from torchvision.ops import roi_pool as tv_roi_pool
            return tv_roi_pool(features, rois, _pair(self.out_size),
                               self.spatial_scale)
        else:
            return roi_pool(features, rois, self.out_size, self.spatial_scale)
