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
    parser.add_argument(
        '--model-options',
        nargs='+',
        help='custom options to special model benchmark')

    args = parser.parse_args()
    return args


basic_arch_root = [
    'atss', 'cascade_rcnn', 'cascade_rpn', 'centripetalnet', 'cornernet',
    'detectors', 'detr', 'double_heads', 'dynamic_rcnn', 'faster_rcnn', 'fcos',
    'foveabox', 'fp16', 'free_anchor', 'fsaf', 'gfl', 'ghm', 'grid_rcnn',
    'guided_anchoring', 'htc', 'libra_rcnn', 'mask_rcnn', 'ms_rcnn',
    'nas_fcos', 'paa', 'pisa', 'point_rend', 'reppoints', 'retinanet', 'rpn',
    'sabl', 'ssd', 'tridentnet', 'vfnet', 'yolact', 'yolo', 'sparse_rcnn',
    'scnet'
]

datasets_root = [
    'wider_face', 'pascal_voc', 'cityscapes', 'lvis', 'deepfashion'
]

data_pipeline_root = ['albu_example', 'instaboost']

nn_module_root = [
    'carafe', 'dcn', 'empirical_attention', 'gcnet', 'gn', 'gn+ws', 'hrnet',
    'pafpn', 'nas_fpn', 'regnet', 'resnest', 'res2net', 'groie'
]

benchmark_pool = [
    'configs/albu_example/mask_rcnn_r50_fpn_albu_1x_coco.py',
    'configs/atss/atss_r50_fpn_1x_coco.py',
    'configs/carafe/mask_rcnn_r50_fpn_carafe_1x_coco.py',
    'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py',
    'configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py',
    'configs/centripetalnet/'
    'centripetalnet_hourglass104_mstest_16x6_210e_coco.py',
    'configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py',
    'configs/cornernet/'
    'cornernet_hourglass104_mstest_8x6_210e_coco.py',  # special
    'configs/dcn/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco.py',
    'configs/dcn/faster_rcnn_r50_fpn_dpool_1x_coco.py',
    'configs/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco.py',
    'configs/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py',
    'configs/detectors/detectors_htc_r50_1x_coco.py',
    'configs/detr/detr_r50_8x2_150e_coco.py',
    'configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py',
    'configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x.py',
    'configs/empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco.py',  # noqa
    'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
    'configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py',
    'configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco.py',
    'configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py',
    'configs/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco.py',
    'configs/fcos/fcos_center_r50_caffe_fpn_gn-head_4x4_1x_coco.py',
    'configs/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py',
    'configs/fp16/retinanet_r50_fpn_fp16_1x_coco.py',
    'configs/fp16/mask_rcnn_r50_fpn_fp16_1x_coco.py',
    'configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py',
    'configs/fsaf/fsaf_r50_fpn_1x_coco.py',
    'configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py',
    'configs/gfl/gfl_r50_fpn_1x_coco.py',
    'configs/ghm/retinanet_ghm_r50_fpn_1x_coco.py',
    'configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py',
    'configs/gn+ws/mask_rcnn_r50_fpn_gn_ws-all_2x_coco.py',
    'configs/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py',
    'configs/groie/faster_rcnn_r50_fpn_groie_1x_coco.py',
    'configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py',
    'configs/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco.py',
    'configs/htc/htc_r50_fpn_1x_coco.py',
    'configs/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco.py',
    'configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py',
    'configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py',
    'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py',
    'configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py',
    'configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py',
    'configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py',
    'configs/paa/paa_r50_fpn_1x_coco.py',
    'configs/pafpn/faster_rcnn_r50_pafpn_1x_coco.py',
    'configs/pisa/pisa_mask_rcnn_r50_fpn_1x_coco.py',
    'configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py',
    'configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py',
    'configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py',
    'configs/res2net/faster_rcnn_r2_101_fpn_2x_coco.py',
    'configs/resnest/'
    'mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py',
    'configs/retinanet/retinanet_r50_caffe_fpn_1x_coco.py',
    'configs/rpn/rpn_r50_fpn_1x_coco.py',
    'configs/sabl/sabl_retinanet_r50_fpn_1x_coco.py',
    'configs/ssd/ssd300_coco.py',
    'configs/tridentnet/tridentnet_r50_caffe_1x_coco.py',
    'configs/vfnet/vfnet_r50_fpn_1x_coco.py',
    'configs/yolact/yolact_r50_1x8_coco.py',
    'configs/yolo/yolov3_d53_320_273e_coco.py',
    'configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py',
    'configs/scnet/scnet_r50_fpn_1x_coco.py'
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

    special_model = args.model_options
    if special_model is not None:
        benchmark_type += special_model

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
