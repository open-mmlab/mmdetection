import argparse
import os
import os.path as osp

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description='Filter configs to train')
    parser.add_argument(
        '--basic-arch',
        action='store_true',
        help='to train models in basic arch')
    parser.add_argument(
        '--datasets', action='store_true', help='to train models in dataset')
    parser.add_argument(
        '--data-pipeline',
        action='store_true',
        help='to train models related to data pipeline, e.g. augmentations')
    parser.add_argument(
        '--nn-module',
        action='store_true',
        help='to train models related to neural network modules')

    args = parser.parse_args()
    return args


basic_arch_root = [
    'cascade_rcnn', 'double_heads', 'fcos', 'foveabox', 'free_anchor',
    'grid_rcnn', 'guided_anchoring', 'htc', 'libra_rcnn', 'atss', 'mask_rcnn',
    'ms_rcnn', 'nas_fpn', 'reppoints', 'retinanet', 'ssd', 'gn', 'ghm', 'fsaf',
    'point_rend', 'nas_fcos', 'pisa', 'dynamic_rcnn'
]

datasets_root = ['wider_face', 'pascal_voc', 'cityscapes', 'mask_rcnn']

data_pipeline_root = [
    'albu_example', 'instaboost', 'ssd', 'mask_rcnn', 'nas_fpn'
]

nn_module_root = [
    'carafe', 'dcn', 'empirical_attention', 'gcnet', 'gn+ws', 'hrnet', 'pafpn',
    'nas_fpn', 'regnet'
]

benchmark_pool = [
    'configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py',
    'configs/htc/htc_r50_fpn_1x_coco.py',
    'configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py',
    'configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py',
    'configs/pisa/pisa_mask_rcnn_r50_fpn_1x_coco.py',
    'configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x.py',
    'configs/ghm/retinanet_ghm_r50_fpn_1x_coco.py',
    'configs/regnet/mask_rcnn_regnetx-3GF_fpn_1x_coco.py',
    'configs/carafe/mask_rcnn_r50_fpn_carafe_1x_coco.py',
    'configs/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py',
    'configs/albu_example/mask_rcnn_r50_fpn_albu_1x_coco.py',
    'configs/rpn/rpn_r50_fpn_1x_coco.py',
    'configs/dcn/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco.py',
    'configs/dcn/faster_rcnn_r50_fpn_dpool_1x_coco.py',
    'configs/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco.py',
    'configs/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py',
    'configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py',
    'configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py',
    'configs/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco.py',
    'configs/retinanet/retinanet_r50_caffe_fpn_1x_coco.py',
    'configs/ssd/ssd300_coco.py',
    'configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py',
    'configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco.py',
    'configs/empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco.py',  # noqa
    'configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py',
    'configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py',
    'configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py',
    'configs/fsaf/fsaf_r50_fpn_1x_coco.py',
    'configs/scratch/mask_rcnn_r50_fpn_gn-all_scratch_6x_coco.py',
    'configs/pafpn/faster_rcnn_r50_pafpn_1x_coco.py',
    'configs/fp16/retinanet_r50_fpn_fp16_1x_coco.py',
    'configs/fp16/mask_rcnn_r50_fpn_fp16_1x_coco.py',
    'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py',
    'configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py',
    'configs/wider_face/ssd300_wider_face.py',
    'configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py',
    'configs/fcos/fcos_center_r50_caffe_fpn_gn-head_4x4_1x_coco.py',
    'configs/atss/atss_r50_fpn_1x_coco.py',
    'configs/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco.py',
    'configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py',
    'configs/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py',
    'configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py',
    'configs/pascal_voc/ssd300_voc0712.py',
    'configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py',
    'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py',
    'configs/gn+ws/mask_rcnn_r50_fpn_gn_ws-all_2x_coco.py'
]


def main():
    args = parse_args()

    benchmark_type = []
    if args.basic_arch:
        benchmark_type += basic_arch_root
    if args.datasets:
        benchmark_type += datasets_root
    if args.data_pipeline:
        benchmark_type += data_pipeline_root
    if args.nn_module:
        benchmark_type += nn_module_root

    config_dpath = 'configs/'
    benchmark_configs = []
    for cfg_root in benchmark_type:
        cfg_dir = osp.join(config_dpath, cfg_root)
        configs = os.scandir(cfg_dir)
        for cfg in configs:
            config_path = osp.join(cfg_dir, cfg.name)
            if (config_path in benchmark_pool
                    and config_path not in benchmark_configs):
                benchmark_configs.append(config_path)

    print(f'Totally found {len(benchmark_configs)} configs to benchmark')
    config_dicts = dict(models=benchmark_configs)
    mmcv.dump(config_dicts, 'regression_test_configs.json')


if __name__ == '__main__':
    main()
