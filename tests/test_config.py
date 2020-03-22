from os.path import dirname, exists, join

from mmcv import Config


def _get_config_directory():
    """ Find the predefined detector config directory """
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(__file__))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet
        repo_dpath = dirname(dirname(mmdet.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def test_config_build_detector():
    """
    Test that all detection models defined in the configs can be initialized.
    """
    from mmdet.models import build_detector

    config_dpath = _get_config_directory()
    print('Found config_dpath = {!r}'.format(config_dpath))

    # import glob
    # config_fpaths = list(glob.glob(join(config_dpath, '**', '*.py')))
    # config_names = [relpath(p, config_dpath) for p in config_fpaths]

    # Only tests a representative subset of configurations

    config_names = [
        # 'dcn/faster_rcnn_dconv_c3-c5_r50_fpn_1x_coco.py',
        # 'dcn/cascade_mask_rcnn_dconv_c3-c5_r50_fpn_1x_coco.py',
        # 'dcn/faster_rcnn_dpool_r50_fpn_1x_coco.py',
        'dcn/mask_rcnn_dconv_c3-c5_r50_fpn_1x_coco.py',
        # 'dcn/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_coco.py',
        # 'dcn/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_coco.py',
        # 'dcn/faster_rcnn_mdpool_r50_fpn_1x_coco.py',
        # 'dcn/faster_rcnn_mdconv_c3-c5_group4_r50_fpn_1x_coco.py',
        # 'dcn/faster_rcnn_mdconv_c3-c5_r50_fpn_1x_coco.py',
        # ---
        # 'htc/htc_x101_32x4d_fpn_20e_16gpu_coco.py',
        'htc/htc_without_semantic_r50_fpn_1x_coco.py',
        # 'htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_coco.py',
        # 'htc/htc_x101_64x4d_fpn_20e_16gpu_coco.py',
        # 'htc/htc_r50_fpn_1x_coco.py',
        # 'htc/htc_r101_fpn_20e.py',
        # 'htc/htc_r50_fpn_20e.py',
        # ---
        'cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py',
        # 'cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py',
        # ---
        # 'scratch/scratch_faster_rcnn_r50_fpn_gn_6x_coco.py',
        # 'scratch/scratch_mask_rcnn_r50_fpn_gn_6x_coco.py',
        # ---
        # 'grid_rcnn/grid_rcnn_gn_head_x101_32x4d_fpn_2x_coco.py',
        'grid_rcnn/grid_rcnn_gn_head_r50_fpn_2x_coco.py',
        # ---
        'double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py',
        # ---
        'empirical_attention/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco'
        '.py',
        # 'empirical_attention/faster_rcnn_r50_fpn_attention_1111_1x_coco.py',
        # 'empirical_attention/faster_rcnn_r50_fpn_attention_0010_1x_coco.py',
        # 'empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco
        # .py',
        # ---
        # 'ms_rcnn/ms_rcnn_r101_caffe_fpn_1x_coco.py',
        # 'ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco.py',
        # 'ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py',
        # ---
        # 'guided_anchoring/ga_faster_x101_32x4d_fpn_1x_coco.py',
        # 'guided_anchoring/ga_rpn_x101_32x4d_fpn_1x_coco.py',
        # 'guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco.py',
        # 'guided_anchoring/ga_fast_r50_caffe_fpn_1x_coco.py',
        # 'guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x_coco.py',
        # 'guided_anchoring/ga_rpn_r101_caffe_rpn_1x_coco.py',
        # 'guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py',
        'guided_anchoring/ga_rpn_r50_caffe_fpn_1x_coco.py',
        # ---
        'foveabox/fovea_r50_fpn_4gpu_1x_coco.py',
        # 'foveabox/fovea_align_gn_ms_r101_fpn_4gpu_2x_coco.py',
        # 'foveabox/fovea_align_gn_r50_fpn_4gpu_2x_coco.py',
        # 'foveabox/fovea_align_gn_r101_fpn_4gpu_2x_coco.py',
        'foveabox/fovea_align_gn_ms_r50_fpn_4gpu_2x_coco.py',
        # ---
        # 'hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco.py',
        # 'hrnet/mask_rcnn_hrnetv2p_w32_1x_coco.py',
        # 'hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py',
        # 'hrnet/htc_hrnetv2p_w32_20e_coco.py',
        # 'hrnet/faster_rcnn_hrnetv2p_w18_1x_coco.py',
        # 'hrnet/mask_rcnn_hrnetv2p_w18_1x_coco.py',
        # 'hrnet/faster_rcnn_hrnetv2p_w32_1x_coco.py',
        # 'hrnet/faster_rcnn_hrnetv2p_w40_1x_coco.py',
        'hrnet/fcos_hrnetv2p_w32_gn_1x_4gpu_coco.py',
        # ---
        # 'gn+ws/faster_rcnn_r50_fpn_gn_ws_1x_coco.py',
        # 'gn+ws/mask_rcnn_x101_32x4d_fpn_gn_ws_2x_coco.py',
        'gn+ws/mask_rcnn_r50_fpn_gn_ws_2x_coco.py',
        # 'gn+ws/mask_rcnn_r50_fpn_gn_ws_20_23_24e_coco.py',
        # ---
        # 'wider_face/ssd300_wider_face.py',
        # ---
        'pascal_voc/ssd300_voc.py',
        'pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py',
        'pascal_voc/ssd512_voc.py',
        # ---
        # 'gcnet/mask_rcnn_r4_gcb_c3-c5_r50_fpn_syncbn_1x_coco.py',
        # 'gcnet/mask_rcnn_r16_gcb_c3-c5_r50_fpn_syncbn_1x_coco.py',
        # 'gcnet/mask_rcnn_r4_gcb_c3-c5_r50_fpn_1x_coco.py',
        # 'gcnet/mask_rcnn_r16_gcb_c3-c5_r50_fpn_1x_coco.py',
        'gcnet/mask_rcnn_r50_fpn_sbn_1x_coco.py',
        # ---
        'gn/mask_rcnn_r50_fpn_gn_contrib_2x_coco.py',
        # 'gn/mask_rcnn_r50_fpn_gn_2x_coco.py',
        # 'gn/mask_rcnn_r101_fpn_gn_2x_coco.py',
        # ---
        # 'reppoints/reppoints_moment_x101_dcn_fpn_2x.py',
        'reppoints/reppoints_moment_r50_fpn_2x_coco.py',
        # 'reppoints/reppoints_moment_x101_dcn_fpn_2x_mt_coco.py',
        'reppoints/reppoints_partial_minmax_r50_fpn_1x_coco.py',
        'reppoints/bbox_r50_grid_center_fpn_1x_coco.py',
        # 'reppoints/reppoints_moment_r101_dcn_fpn_2x_coco.py',
        # 'reppoints/reppoints_moment_r101_fpn_2x_mt_coco.py',
        # 'reppoints/reppoints_moment_r50_fpn_2x_mt_coco.py',
        'reppoints/reppoints_minmax_r50_fpn_1x_coco.py',
        # 'reppoints/reppoints_moment_r50_fpn_1x_coco.py',
        # 'reppoints/reppoints_moment_r101_fpn_2x_coco.py',
        # 'reppoints/reppoints_moment_r101_dcn_fpn_2x_mt_coco.py',
        'reppoints/bbox_r50_grid_fpn_1x_coco.py',
        # ---
        # 'fcos/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x_coco.py',
        # 'fcos/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu_coco.py',
        'fcos/fcos_r50_caffe_fpn_gn_1x_4gpu_coco.py',
        # ---
        'albu_example/mask_rcnn_r50_fpn_1x_albu_coco.py',
        # ---
        'libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py',
        # 'libra_rcnn/libra_retinanet_r50_fpn_1x_coco.py',
        # 'libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco.py',
        # 'libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco.py',
        # 'libra_rcnn/libra_fast_rcnn_r50_fpn_1x_coco.py',
        # ---
        # 'ghm/retinanet_ghm_r50_fpn_1x_coco.py',
        # ---
        # 'fp16/retinanet_r50_fpn_fp16_1x_coco.py',
        'fp16/mask_rcnn_r50_fpn_fp16_1x_coco.py',
        'fp16/faster_rcnn_r50_fpn_fp16_1x_coco.py'
    ]

    print('Using {} config files'.format(len(config_names)))

    for config_fname in config_names:
        config_fpath = join(config_dpath, config_fname)
        config_mod = Config.fromfile(config_fpath)

        config_mod.model
        config_mod.train_cfg
        config_mod.test_cfg
        print('Building detector, config_fpath = {!r}'.format(config_fpath))

        # Remove pretrained keys to allow for testing in an offline environment
        if 'pretrained' in config_mod.model:
            config_mod.model['pretrained'] = None

        detector = build_detector(
            config_mod.model,
            train_cfg=config_mod.train_cfg,
            test_cfg=config_mod.test_cfg)
        assert detector is not None
