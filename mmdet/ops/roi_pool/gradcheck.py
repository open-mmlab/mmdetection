import os.path as osp
import sys

import torch
from torch.autograd import gradcheck

sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from roi_pool import RoIPool  # noqa: E402, isort:skip

feat = torch.randn(4, 16, 15, 15, requires_grad=True).cuda()
rois = torch.Tensor([[0, 0, 0, 50, 50], [0, 10, 30, 43, 55],
                     [1, 67, 40, 110, 120]]).cuda()
inputs = (feat, rois)
print('Gradcheck for roi pooling...')
test = gradcheck(RoIPool(4, 1.0 / 8), inputs, eps=1e-5, atol=1e-3)
print(test)
