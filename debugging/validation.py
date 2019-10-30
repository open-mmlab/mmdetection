"""Validation.

Allows direct access to validation loops and allows debugging of those loops.
Note: This runs on CPU and not CUDA since we want to access the values and
manipulate them more easily

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from torch.utils.data import DataLoader

from mmcv import Config

from mmdet.models.anchor_heads import WFCOSHead
from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN

from debugging.multi_transforms import *
from debugging.coco_dataset import CocoDataset


class ValidationDebug:
    def __init__(self, config_path):
        """Initializes the network and dataset."""
        cfg = Config.fromfile(config_path)
        self.cfg = cfg

        # Initialize network first as separate modules so we can access WFCOS
        self.backbone = ResNet(**cfg.model.backbone)
        self.neck = FPN(**cfg.model.neck)
        self.head = WFCOSHead(**cfg.model.bbox_head)

        # Now make the dataloader
        transforms = MultiCompose([
            MultiResize((640, 800)),
            MultiToTensor(),
            MultiNormalize(**cfg.img_norm_cfg)
        ])
        coco_dataset = CocoDataset(cfg.data.train.img_prefix,
                                   cfg.data.train.ann_file,
                                   transforms=transforms)

        self.loader = DataLoader(coco_dataset, 1, True)

    def run(self):
        """Runs validation only with a completely untrained network."""
        for i, (img, cls) in enumerate(self.loader):
            out = self.backbone(img)
            out = self.neck(out)
            out = self.head(out)

            img_metas = [{'img_shape': (640, 800),
                          'scale_factor': 1}]
            self.head.get_bboxes(out[0], out[1], out[2], img_metas,
                                 self.cfg.test_cfg)
            pass


if __name__ == '__main__':
    vd = ValidationDebug('configs/wfcos/wfcos_validation.py')
    vd.run()
