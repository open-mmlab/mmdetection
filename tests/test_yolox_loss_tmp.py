import torch
import mmcv
from mmdet.models import build_detector



def test_yolox_loss():
    cfg = 'configs/yolox/yolox_tiny.py'
    config = mmcv.Config.fromfile(cfg)

    model = build_detector(config.model)

    for i in range(13):

        loss_input = torch.load(f'work_dirs/yolox/test_loss/dumped_obj/loss_input{i*50}.pth')

        loss_output = torch.load(f'work_dirs/yolox/test_loss/dumped_obj/loss_output{i*50}.pth')

        new_loss_out = model.bbox_head.loss(*loss_input)

        a = 1

