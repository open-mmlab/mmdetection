import numpy as np
import torch
from torch.autograd import gradcheck

import os.path as osp
import sys
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from roi_align import RoIAlign  # noqa: E402

feat_size = 15
spatial_scale = 1.0 / 8
img_size = feat_size / spatial_scale
num_imgs = 2
num_rois = 20

batch_ind = np.random.randint(num_imgs, size=(num_rois, 1))
rois = np.random.rand(num_rois, 4) * img_size * 0.5
rois[:, 2:] += img_size * 0.5
rois = np.hstack((batch_ind, rois))

feat = torch.randn(
    num_imgs, 16, feat_size, feat_size, requires_grad=True, device='cuda:0')
rois = torch.from_numpy(rois).float().cuda()
inputs = (feat, rois)
print('Gradcheck for roi align...')
test = gradcheck(RoIAlign(3, spatial_scale), inputs, atol=1e-3, eps=1e-3)
print(test)
test = gradcheck(RoIAlign(3, spatial_scale, 2), inputs, atol=1e-3, eps=1e-3)
print(test)
