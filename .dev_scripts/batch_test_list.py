# Copyright (c) OpenMMLab. All rights reserved.

# missing wider_face/timm_example/strong_baselines/simple_copy_paste/
# selfsup_pretrain/seesaw_loss/pascal_voc/openimages/lvis/ld/lad/cityscapes/deepfashion

# yapf: disable
atss = dict(
    config='configs/atss/atss_r50_fpn_1x_coco.py',
    checkpoint='atss_r50_fpn_1x_coco_20200209-985f7bd0.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=39.4),
)
autoassign = dict(
    config='configs/autoassign/autoassign_r50-caffe_fpn_1x_coco.py',
    checkpoint='auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/autoassign/auto_assign_r50_fpn_1x_coco/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=40.4),
)
carafe = dict(
    config='configs/carafe/faster-rcnn_r50_fpn-carafe_1x_coco.py',
    checkpoint='faster_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.386_20200504_175733-385a75b7.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/carafe/faster_rcnn_r50_fpn_carafe_1x_coco/faster_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.386_20200504_175733-385a75b7.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=38.6),
)
cascade_rcnn = [
    dict(
        config='configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py',
        checkpoint='cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth',
        eval='bbox',
        url='https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth', # noqa
        metric=dict(bbox_mAP=40.3),
    ),
    dict(
        config='configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py',
        checkpoint='cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth',
        url='https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth', # noqa
        eval=['bbox', 'segm'],
        metric=dict(bbox_mAP=41.2, segm_mAP=35.9),
    ),
]
cascade_rpn = dict(
    config='configs/cascade_rpn/cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco.py', # noqa
    checkpoint='crpn_faster_rcnn_r50_caffe_fpn_1x_coco-c8283cca.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco/crpn_faster_rcnn_r50_caffe_fpn_1x_coco-c8283cca.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=40.4),
)
centernet = dict(
    config='configs/centernet/centernet_r18-dcnv2_8xb16-crop512-140e_coco.py',
    checkpoint='centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth', # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=29.5),
)
centripetalnet = dict(
    config='configs/centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py',  # noqa
    checkpoint='centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804-3ccc61e5.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco/centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804-3ccc61e5.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=44.7),
)
convnext = dict(
    config='configs/convnext/cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py', # noqa
    checkpoint='cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220510_201004-3d24f5a4.pth', # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220510_201004-3d24f5a4.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=51.8, segm_mAP=44.8),
)
cornernet = dict(
    config='configs/cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py',
    checkpoint='cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618-79b44c30.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco/cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618-79b44c30.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=41.2),
)
dcn = dict(
    config='configs/dcn/faster-rcnn_r50-dconv-c3-c5_fpn_1x_coco.py',
    checkpoint='faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=41.3),
)
dcnv2 = dict(
    config='configs/dcnv2/faster-rcnn_r50_fpn_mdpool_1x_coco.py',
    checkpoint='faster_rcnn_r50_fpn_mdpool_1x_coco_20200307-c0df27ff.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco/faster_rcnn_r50_fpn_mdpool_1x_coco_20200307-c0df27ff.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=38.7),
)
ddod = dict(
    config='configs/ddod/ddod_r50_fpn_1x_coco.py',
    checkpoint='ddod_r50_fpn_1x_coco_20220523_223737-29b2fc67.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/ddod/ddod_r50_fpn_1x_coco/ddod_r50_fpn_1x_coco_20220523_223737-29b2fc67.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=41.7),
)
deformable_detr = dict(
    config='configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py',
    checkpoint='deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=44.5),
)
detectors = dict(
    config='configs/detectors/detectors_htc-r50_1x_coco.py',
    checkpoint='detectors_htc_r50_1x_coco-329b1453.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco-329b1453.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=49.1, segm_mAP=42.6),
)
detr = dict(
    config='configs/detr/detr_r50_8xb2-150e_coco.py',
    checkpoint='detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=40.1),
)
double_heads = dict(
    config='configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py',
    checkpoint='dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/double_heads/dh_faster_rcnn_r50_fpn_1x_coco/dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=40.0),
)
dyhead = dict(
    config='configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py',
    checkpoint='atss_r50_fpn_dyhead_4x4_1x_coco_20211219_023314-eaa620c6.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/dyhead/atss_r50_fpn_dyhead_4x4_1x_coco/atss_r50_fpn_dyhead_4x4_1x_coco_20211219_023314-eaa620c6.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=43.3),
)
dynamic_rcnn = dict(
    config='configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py',
    checkpoint='dynamic_rcnn_r50_fpn_1x-62a3f276.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x/dynamic_rcnn_r50_fpn_1x-62a3f276.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=38.9),
)
efficientnet = dict(
    config='configs/efficientnet/retinanet_effb3_fpn_8xb4-crop896-1x_coco.py',
    checkpoint='retinanet_effb3_fpn_crop896_8x4_1x_coco_20220322_234806-615a0dda.pth', # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/efficientnet/retinanet_effb3_fpn_crop896_8x4_1x_coco/retinanet_effb3_fpn_crop896_8x4_1x_coco_20220322_234806-615a0dda.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=40.5),
)
empirical_attention = dict(
    config='configs/empirical_attention/faster-rcnn_r50-attn1111_fpn_1x_coco.py',  # noqa
    checkpoint='faster_rcnn_r50_fpn_attention_1111_1x_coco_20200130-403cccba.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_1x_coco/faster_rcnn_r50_fpn_attention_1111_1x_coco_20200130-403cccba.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=40.0),
)
faster_rcnn = dict(
    config='configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
    checkpoint='faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=37.4),
)
fcos = dict(
    config='configs/fcos/fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py',  # noqa
    checkpoint='fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=38.7),
)
foveabox = dict(
    config='configs/foveabox/fovea_r50_fpn_gn-head-align_4xb4-2x_coco.py',
    checkpoint='fovea_align_r50_fpn_gn-head_4x4_2x_coco_20200203-8987880d.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco/fovea_align_r50_fpn_gn-head_4x4_2x_coco_20200203-8987880d.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=37.9),
)
fpg = dict(
    config='configs/fpg/mask-rcnn_r50_fpg_crop640-50e_coco.py',
    checkpoint='mask_rcnn_r50_fpg_crop640_50e_coco_20220311_011857-233b8334.pth', # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg_crop640_50e_coco/mask_rcnn_r50_fpg_crop640_50e_coco_20220311_011857-233b8334.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=43.0, segm_mAP=38.1),
)
free_anchor = dict(
    config='configs/free_anchor/freeanchor_r50_fpn_1x_coco.py',
    checkpoint='retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=38.7),
)
fsaf = dict(
    config='configs/fsaf/fsaf_r50_fpn_1x_coco.py',
    checkpoint='fsaf_r50_fpn_1x_coco-94ccc51f.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco-94ccc51f.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=37.4),
)
gcnet = dict(
    config='configs/gcnet/mask-rcnn_r50-syncbn-gcb-r16-c3-c5_fpn_1x_coco.py',  # noqa
    checkpoint='mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200202-587b99aa.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200202-587b99aa.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=40.4, segm_mAP=36.2),
)
gfl = dict(
    config='configs/gfl/gfl_r50_fpn_1x_coco.py',
    checkpoint='gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=40.2),
)
ghm = dict(
    config='configs/ghm/retinanet_r50_fpn_ghm-1x_coco.py',
    checkpoint='retinanet_ghm_r50_fpn_1x_coco_20200130-a437fda3.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_r50_fpn_1x_coco/retinanet_ghm_r50_fpn_1x_coco_20200130-a437fda3.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=37.0),
)
gn = dict(
    config='configs/gn/mask-rcnn_r50_fpn_gn-all_2x_coco.py',
    checkpoint='mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_2x_coco/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=40.1, segm_mAP=36.4),
)
gn_ws = dict(
    config='configs/gn+ws/faster-rcnn_r50_fpn_gn-ws-all_1x_coco.py',
    checkpoint='faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=39.7),
)
grid_rcnn = dict(
    config='configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_coco.py',
    checkpoint='grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=40.4),
)
groie = dict(
    config='configs/groie/faste-rcnn_r50_fpn_groie_1x_coco.py',
    checkpoint='faster_rcnn_r50_fpn_groie_1x_coco_20200604_211715-66ee9516.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/groie/faster_rcnn_r50_fpn_groie_1x_coco/faster_rcnn_r50_fpn_groie_1x_coco_20200604_211715-66ee9516.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=38.3),
)
guided_anchoring = dict(
        config='configs/guided_anchoring/ga-retinanet_r50-caffe_fpn_1x_coco.py',  # noqa
        checkpoint='ga_retinanet_r50_caffe_fpn_1x_coco_20201020-39581c6f.pth',
        url='https://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco/ga_retinanet_r50_caffe_fpn_1x_coco_20201020-39581c6f.pth', # noqa
        eval='bbox',
        metric=dict(bbox_mAP=36.9),
    )
hrnet = dict(
    config='configs/hrnet/faster-rcnn_hrnetv2p-w18-1x_coco.py',
    checkpoint='faster_rcnn_hrnetv2p_w18_1x_coco_20200130-56651a6d.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco/faster_rcnn_hrnetv2p_w18_1x_coco_20200130-56651a6d.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=36.9),
)
htc = dict(
    config='configs/htc/htc_r50_fpn_1x_coco.py',
    checkpoint='htc_r50_fpn_1x_coco_20200317-7332cf16.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_1x_coco/htc_r50_fpn_1x_coco_20200317-7332cf16.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=42.3, segm_mAP=37.4),
)
instaboost = dict(
    config='configs/instaboost/mask-rcnn_r50_fpn_instaboost-4x_coco.py',
    checkpoint='mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-d025f83a.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco/mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-d025f83a.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=40.6, segm_mAP=36.6),
)
libra_rcnn = dict(
    config='configs/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py',
    checkpoint='libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=38.3),
)
mask2former = dict(
    config='configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py',
    checkpoint='mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth', # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth', # noqa
    eval=['bbox', 'segm', 'PQ'],
    metric=dict(PQ=51.9, bbox_mAP=44.8, segm_mAP=41.9),
)
mask_rcnn = dict(
    config='configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py',
    checkpoint='mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=38.2, segm_mAP=34.7),
)
maskformer = dict(
    config='configs/maskformer/maskformer_r50_ms-16xb1-75e_coco.py',
    checkpoint='maskformer_r50_mstrain_16x1_75e_coco_20220221_141956-bc2699cb.pth', # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/maskformer/maskformer_r50_mstrain_16x1_75e_coco/maskformer_r50_mstrain_16x1_75e_coco_20220221_141956-bc2699cb.pth', # noqa
    eval='PQ',
    metric=dict(PQ=46.9),
)
ms_rcnn = dict(
    config='configs/ms_rcnn/ms-rcnn_r50-caffe_fpn_1x_coco.py',
    checkpoint='ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=38.2, segm_mAP=36.0),
)
nas_fcos = dict(
    config='configs/nas_fcos/nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco.py',  # noqa
    checkpoint='nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200520-1bdba3ce.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200520-1bdba3ce.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=39.4),
)
nas_fpn = dict(
    config='configs/nas_fpn/retinanet_r50_nasfpn_crop640-50e_coco.py',
    checkpoint='retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=40.5),
)
paa = dict(
    config='configs/paa/paa_r50_fpn_1x_coco.py',
    checkpoint='paa_r50_fpn_1x_coco_20200821-936edec3.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=40.4),
)
pafpn = dict(
    config='configs/pafpn/faster-rcnn_r50_pafpn_1x_coco.py',
    checkpoint='faster_rcnn_r50_pafpn_1x_coco_bbox_mAP-0.375_20200503_105836-b7b4b9bd.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/pafpn/faster_rcnn_r50_pafpn_1x_coco/faster_rcnn_r50_pafpn_1x_coco_bbox_mAP-0.375_20200503_105836-b7b4b9bd.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=37.5),
)
panoptic_fpn = dict(
    config='configs/panoptic_fpn/panoptic-fpn_r50_fpn_1x_coco.py',
    checkpoint='panoptic_fpn_r50_fpn_1x_coco_20210821_101153-9668fd13.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco/panoptic_fpn_r50_fpn_1x_coco_20210821_101153-9668fd13.pth', # noqa
    eval='PQ',
    metric=dict(PQ=40.2),
)
pisa = dict(
    config='configs/pisa/faster-rcnn_r50_fpn_pisa_1x_coco.py',
    checkpoint='pisa_faster_rcnn_r50_fpn_1x_coco-dea93523.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_r50_fpn_1x_coco/pisa_faster_rcnn_r50_fpn_1x_coco-dea93523.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=38.4),
)
point_rend = dict(
    config='configs/point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py',
    checkpoint='point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=38.4, segm_mAP=36.3),
)
pvt = dict(
    config='configs/pvt/retinanet_pvt-s_fpn_1x_coco.py',
    checkpoint='retinanet_pvt-s_fpn_1x_coco_20210906_142921-b6c94a5b.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvt-s_fpn_1x_coco/retinanet_pvt-s_fpn_1x_coco_20210906_142921-b6c94a5b.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=40.4),
)
queryinst = dict(
    config='configs/queryinst/queryinst_r50_fpn_1x_coco.py',
    checkpoint='queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_1x_coco/queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=42.0, segm_mAP=37.5),
)
regnet = dict(
    config='configs/regnet/mask-rcnn_regnetx-3.2GF_fpn_1x_coco.py',
    checkpoint='mask_rcnn_regnetx-3.2GF_fpn_1x_coco_20200520_163141-2a9d1814.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco/mask_rcnn_regnetx-3.2GF_fpn_1x_coco_20200520_163141-2a9d1814.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=40.4, segm_mAP=36.7),
)
reppoints = dict(
    config='configs/reppoints/reppoints-moment_r50_fpn_1x_coco.py',
    checkpoint='reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_1x_coco/reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=37.0),
)
res2net = dict(
    config='configs/res2net/faster-rcnn_res2net-101_fpn_2x_coco.py',
    checkpoint='faster_rcnn_r2_101_fpn_2x_coco-175f1da6.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/res2net/faster_rcnn_r2_101_fpn_2x_coco/faster_rcnn_r2_101_fpn_2x_coco-175f1da6.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=43.0),
)
resnest = dict(
    config='configs/resnest/faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco.py',  # noqa
    checkpoint='faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco_20200926_125502-20289c16.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20200926_125502-20289c16.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=42.0),
)
resnet_strikes_back = dict(
    config='configs/resnet_strikes_back/mask-rcnn_r50-rsb-pre_fpn_1x_coco.py', # noqa
    checkpoint='mask_rcnn_r50_fpn_rsb-pretrain_1x_coco_20220113_174054-06ce8ba0.pth', # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/resnet_strikes_back/mask_rcnn_r50_fpn_rsb-pretrain_1x_coco/mask_rcnn_r50_fpn_rsb-pretrain_1x_coco_20220113_174054-06ce8ba0.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=41.2, segm_mAP=38.2),
)
retinanet = dict(
    config='configs/retinanet/retinanet_r50_fpn_1x_coco.py',
    checkpoint='retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=36.5),
)
rpn = dict(
    config='configs/rpn/rpn_r50_fpn_1x_coco.py',
    checkpoint='rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_1x_coco/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth', # noqa
    eval='proposal_fast',
    metric=dict(AR_1000=58.2),
)
sabl = [
    dict(
        config='configs/sabl/sabl-retinanet_r50_fpn_1x_coco.py',
        checkpoint='sabl_retinanet_r50_fpn_1x_coco-6c54fd4f.pth',
        url='https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_1x_coco/sabl_retinanet_r50_fpn_1x_coco-6c54fd4f.pth', # noqa
        eval='bbox',
        metric=dict(bbox_mAP=37.7),
    ),
    dict(
        config='configs/sabl/sabl-faster-rcnn_r50_fpn_1x_coco.py',
        checkpoint='sabl_faster_rcnn_r50_fpn_1x_coco-e867595b.pth',
        url='https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r50_fpn_1x_coco/sabl_faster_rcnn_r50_fpn_1x_coco-e867595b.pth', # noqa
        eval='bbox',
        metric=dict(bbox_mAP=39.9),
    ),
]
scnet = dict(
    config='configs/scnet/scnet_r50_fpn_1x_coco.py',
    checkpoint='scnet_r50_fpn_1x_coco-c3f09857.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r50_fpn_1x_coco/scnet_r50_fpn_1x_coco-c3f09857.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=43.5),
)
scratch = dict(
    config='configs/scratch/mask-rcnn_r50-scratch_fpn_gn-all_6x_coco.py',
    checkpoint='scratch_mask_rcnn_r50_fpn_gn_6x_bbox_mAP-0.412__segm_mAP-0.374_20200201_193051-1e190a40.pth', # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/scratch/mask_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_mask_rcnn_r50_fpn_gn_6x_bbox_mAP-0.412__segm_mAP-0.374_20200201_193051-1e190a40.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=41.2, segm_mAP=37.4),
)
solo = dict(
    config='configs/solo/decoupled-solo_r50_fpn_1x_coco.py',
    checkpoint='decoupled_solo_r50_fpn_1x_coco_20210820_233348-6337c589.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/solo/decoupled_solo_r50_fpn_1x_coco/decoupled_solo_r50_fpn_1x_coco_20210820_233348-6337c589.pth', # noqa
    eval='segm',
    metric=dict(segm_mAP=33.9),
)
solov2 = dict(
    config='configs/solov2/solov2_r50_fpn_1x_coco.py',
    checkpoint='solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_1x_coco/solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth', # noqa
    eval='segm',
    metric=dict(segm_mAP=34.8),
)
sparse_rcnn = dict(
    config='configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py',
    checkpoint='sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=37.9),
)
ssd = [
    dict(
        config='configs/ssd/ssd300_coco.py',
        checkpoint='ssd300_coco_20210803_015428-d231a06e.pth',
        url='https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth', # noqa
        eval='bbox',
        metric=dict(bbox_mAP=25.5),
    ),
    dict(
        config='configs/ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py',
        checkpoint='ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth',  # noqa
        url='https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth', # noqa
        eval='bbox',
        metric=dict(bbox_mAP=21.3),
    ),
]
swin = dict(
    config='configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py',
    checkpoint='mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth', # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=42.7, segm_mAP=39.3),
)
tood = dict(
    config='configs/tood/tood_r50_fpn_1x_coco.py',
    checkpoint='tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r50_fpn_1x_coco/tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=42.4),
)
tridentnet = dict(
    config='configs/tridentnet/tridentnet_r50-caffe_1x_coco.py',
    checkpoint='tridentnet_r50_caffe_1x_coco_20201230_141838-2ec0b530.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_1x_coco/tridentnet_r50_caffe_1x_coco_20201230_141838-2ec0b530.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=37.6),
)
vfnet = dict(
    config='configs/vfnet/vfnet_r50_fpn_1x_coco.py',
    checkpoint='vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_1x_coco/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=41.6),
)
yolact = dict(
    config='configs/yolact/yolact_r50_1xb8-55e_coco.py',
    checkpoint='yolact_r50_1x8_coco_20200908-f38d58df.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/yolact/yolact_r50_1x8_coco/yolact_r50_1x8_coco_20200908-f38d58df.pth', # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=31.2, segm_mAP=29.0),
)
yolo = dict(
    config='configs/yolo/yolov3_d53_8xb8-320-273e_coco.py',
    checkpoint='yolov3_d53_320_273e_coco-421362b6.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=27.9),
)
yolof = dict(
    config='configs/yolof/yolof_r50-c5_8xb8-1x_coco.py',
    checkpoint='yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth',
    url='https://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=37.5),
)
yolox = dict(
    config='configs/yolox/yolox_tiny_8xb8-300e_coco.py',
    checkpoint='yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=31.8),
)
# yapf: enable
