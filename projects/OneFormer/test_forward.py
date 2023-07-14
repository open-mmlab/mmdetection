import copy
from os.path import dirname, exists, join

import numpy as np
import pytest
import torch


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet
        repo_dpath = dirname(dirname(mmdet.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


@pytest.mark.parametrize('cfg_file', [
    'configs/oneformer/oneformer_r50_lsj_8x2_50e_coco-panoptic.py',
    'configs/oneformer/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco-panoptic.py'
])
def test_oneformer_forward(cfg_file):
    # # Test Panoptic Segmentation and Instance Segmentation
    model_cfg = _get_detector_cfg(cfg_file)
    # base_channels = 32
    # model_cfg.backbone.depth = 18
    # model_cfg.backbone.init_cfg = None
    # model_cfg.backbone.base_channels = base_channels
    # model_cfg.panoptic_head.in_channels = [
    #     base_channels * 2**i for i in range(4)
    # ]
    # model_cfg.panoptic_head.feat_channels = base_channels
    # model_cfg.panoptic_head.out_channels = base_channels
    # model_cfg.panoptic_head.pixel_decoder.encoder.\
    #     transformerlayers.attn_cfgs.embed_dims = base_channels
    # model_cfg.panoptic_head.pixel_decoder.encoder.\
    #     transformerlayers.ffn_cfgs.embed_dims = base_channels
    # model_cfg.panoptic_head.pixel_decoder.encoder.\
    #     transformerlayers.ffn_cfgs.feedforward_channels = base_channels * 4
    # model_cfg.panoptic_head.pixel_decoder.\
    #     positional_encoding.num_feats = base_channels // 2
    # model_cfg.panoptic_head.positional_encoding.\
    #     num_feats = base_channels // 2
    # model_cfg.panoptic_head.transformer_decoder.\
    #     transformerlayers.attn_cfgs.embed_dims = base_channels
    # model_cfg.panoptic_head.transformer_decoder.\
    #     transformerlayers.ffn_cfgs.embed_dims = base_channels
    # model_cfg.panoptic_head.transformer_decoder.\
    #     transformerlayers.ffn_cfgs.feedforward_channels = base_channels * 8
    # model_cfg.panoptic_head.transformer_decoder.\
    #     transformerlayers.feedforward_channels = base_channels * 8

    num_stuff_classes = model_cfg.panoptic_head.num_stuff_classes

    from mmdet.core import BitmapMasks
    from mmdet.models import build_detector
    detector = build_detector(model_cfg)

    def _forward_train():
        losses = detector.forward(
            img,
            img_metas,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            gt_masks=gt_masks,
            gt_semantic_seg=gt_semantic_seg,
            return_loss=True)
        assert isinstance(losses, dict)
        loss, _ = detector._parse_losses(losses)
        assert float(loss.item()) > 0

    # Test forward train with non-empty truth batch
    detector.train()
    img_metas = [
        {
            'batch_input_shape': (128, 160),
            'img_shape': (126, 160, 3),
            'ori_shape': (63, 80, 3),
            'pad_shape': (128, 160, 3)
        },
    ]
    img = torch.rand((1, 3, 128, 160))
    gt_bboxes = None
    gt_labels = [
        torch.tensor([10]).long(),
    ]
    thing_mask1 = np.zeros((1, 128, 160), dtype=np.int32)
    thing_mask1[0, :50] = 1
    gt_masks = [
        BitmapMasks(thing_mask1, 128, 160),
    ]
    stuff_mask1 = torch.zeros((1, 128, 160)).long()
    stuff_mask1[0, :50] = 10
    stuff_mask1[0, 50:] = 100
    gt_semantic_seg = [
        stuff_mask1,
    ]
    _forward_train()

    # Test forward train with non-empty truth batch and gt_semantic_seg=None
    gt_semantic_seg = None
    _forward_train()

    # Test forward train with an empty truth batch
    gt_bboxes = [
        torch.empty((0, 4)).float(),
    ]
    gt_labels = [
        torch.empty((0, )).long(),
    ]
    mask = np.zeros((0, 128, 160), dtype=np.uint8)
    gt_masks = [
        BitmapMasks(mask, 128, 160),
    ]
    gt_semantic_seg = [
        torch.randint(0, 133, (0, 128, 160)),
    ]
    _forward_train()

    # Test forward train with an empty truth batch and gt_semantic_seg=None
    gt_semantic_seg = None
    _forward_train()

    # Test forward test
    detector.eval()
    with torch.no_grad():
        img_list = [g[None, :] for g in img]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      rescale=True,
                                      return_loss=False)

            if num_stuff_classes > 0:
                assert isinstance(result[0], dict)
            else:
                assert isinstance(result[0], tuple)

        batch_results.append(result)
