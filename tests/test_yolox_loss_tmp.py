import torch
import mmcv
from mmdet.models import build_detector



def test_yolox_loss():
    cfg = 'configs/yolox/yolox_s.py'
    config = mmcv.Config.fromfile(cfg)

    model = build_detector(config.model)

    loss_input = torch.load('work_dir/yolox_s_test/dumped_obj/loss_input.pth')

    loss_output = torch.load('work_dir/yolox_s_test/dumped_obj/loss_output.pth')

    new_loss_out = model.bbox_head.loss(*loss_input)

    a = 1

