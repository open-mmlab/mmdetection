import os.path as osp
import sys

import mmcv
import torch
from torch.autograd import gradcheck

sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from mmdet.ops.carafe import CARAFENAIVE  # noqa: E402, isort:skip
from mmdet.ops.carafe import carafe_naive  # noqa: E402, isort:skip
from mmdet.ops.carafe import carafe, CARAFE  # noqa: E402, isort:skip

feat = torch.randn(2, 64, 3, 3, requires_grad=True, device='cuda:0').double()
mask = torch.randn(
    2, 100, 6, 6, requires_grad=True, device='cuda:0').sigmoid().double()

print('Gradcheck for carafe...')
test = gradcheck(CARAFE(5, 4, 2), (feat, mask), atol=1e-4, eps=1e-4)
print(test)

print('Gradcheck for carafe naive...')
test = gradcheck(CARAFENAIVE(5, 4, 2), (feat, mask), atol=1e-4, eps=1e-4)
print(test)

feat = torch.randn(
    2, 1024, 100, 100, requires_grad=True, device='cuda:0').float()
mask = torch.randn(
    2, 25, 200, 200, requires_grad=True, device='cuda:0').sigmoid().float()
loop_num = 500

time_forward = 0
time_backward = 0
bar = mmcv.ProgressBar(loop_num)
timer = mmcv.Timer()
for i in range(loop_num):
    x = carafe(feat.clone(), mask.clone(), 5, 1, 2)
    torch.cuda.synchronize()
    time_forward += timer.since_last_check()
    x.sum().backward(retain_graph=True)
    torch.cuda.synchronize()
    time_backward += timer.since_last_check()
    bar.update()
print('\nCARAFE time forward: {} ms/iter | time backward: {} ms/iter'.format(
    (time_forward + 1e-3) * 1e3 / loop_num,
    (time_backward + 1e-3) * 1e3 / loop_num))

time_naive_forward = 0
time_naive_backward = 0
bar = mmcv.ProgressBar(loop_num)
timer = mmcv.Timer()
for i in range(loop_num):
    x = carafe_naive(feat.clone(), mask.clone(), 5, 1, 2)
    torch.cuda.synchronize()
    time_naive_forward += timer.since_last_check()
    x.sum().backward(retain_graph=True)
    torch.cuda.synchronize()
    time_naive_backward += timer.since_last_check()
    bar.update()
print('\nCARAFE naive time forward: {} ms/iter | time backward: {} ms/iter'.
      format((time_naive_forward + 1e-3) * 1e3 / loop_num,
             (time_naive_backward + 1e-3) * 1e3 / loop_num))
