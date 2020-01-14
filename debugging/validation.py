"""Validation.

Allows direct access to validation loops and allows debugging of those loops.
Note: This runs on CPU and not CUDA since we want to access the values and
manipulate them more easily

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch
from torch.utils.data import DataLoader

from mmcv import Config

from collections import OrderedDict

from mmdet.models.builder import build_backbone, build_neck, build_head

from mmdet.models.anchor_heads import WFCOSHead
from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN

from debugging.multi_transforms import *
from debugging.coco_dataset import CocoDataset

from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser(description='debug the validation method')

    parser.add_argument('CONFIG', type=str, help='configuration file path')

    return parser.parse_args()


class ValidationDebug:
    def __init__(self, config_path):
        """Initializes the network and dataset."""
        cfg = Config.fromfile(config_path)
        self.cfg = cfg

        # Get the checkpoint file
        print('loading checkpoint file ...')
        cp = torch.load(cfg.work_dir + '/latest.pth')
        print('done')

        print('loading state dictionary ...')
        # Initialize network first as separate modules so we can access WFCOS
        self.backbone = build_backbone(**cfg.model.backbone)
        self.neck = build_neck(**cfg.model.neck)
        self.head = build_head(**cfg.model.bbox_head)

        # Load the state dicts
        backbone_state = OrderedDict()
        neck_state = OrderedDict()
        head_state = OrderedDict()

        for key in cp['state_dict'].keys():
            if 'backbone' in key:
                backbone_state[key.split('.', 1)[1]] = cp['state_dict'][key]
            elif 'neck' in key:
                neck_state[key.split('.', 1)[1]] = cp['state_dict'][key]
            elif 'bbox_head' in key:
                head_state[key.split('.', 1)[1]] = cp['state_dict'][key]

        self.backbone.load_state_dict(backbone_state)
        self.neck.load_state_dict(neck_state)
        self.head.load_state_dict(head_state)

        print('done')

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
        """Runs validation only with the network."""
        print('starting inference validation run ...')
        for i, (img, cls) in enumerate(self.loader):
            out = self.backbone(img)
            out = self.neck(out)
            out = self.head(out)

            img_metas = [{'img_shape': (640, 800),
                          'scale_factor': 1}]
            bboxes = self.head.get_bboxes(out[0], out[1], out[2], img_metas,
                                          self.cfg.test_cfg)
            pass
        print('done')


if __name__ == '__main__':
    args = parse_arguments()
    vd = ValidationDebug(args.CONFIG)
    vd.run()
