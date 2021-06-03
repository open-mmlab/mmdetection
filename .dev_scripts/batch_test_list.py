# yapf: disable
atss = dict(
    config='configs/atss/atss_r50_fpn_1x_coco.py',
    checkpoint='atss_r50_fpn_1x_coco_20200209-985f7bd0.pth',
    eval='bbox',
    metric=dict(bbox_mAP=39.4),
)
autoassign = dict(
    config='configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py',
    checkpoint='auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth',
    eval='bbox',
    metric=dict(bbox_mAP=40.4),
)
carafe = dict(
    config='configs/carafe/faster_rcnn_r50_fpn_carafe_1x_coco.py',
    checkpoint='faster_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.386_20200504_175733-385a75b7.pth',  # noqa
    eval='bbox',
    metric=dict(bbox_mAP=38.6),
)
cascade_rcnn = [
    dict(
        config='configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py',
        checkpoint='cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth',
        eval='bbox',
        metric=dict(bbox_mAP=40.3),
    ),
    dict(
        config='configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py',
        checkpoint='cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth',
        eval=['bbox', 'segm'],
        metric=dict(bbox_mAP=41.2, segm_mAP=35.9),
    ),
]
cascade_rpn = dict(
    config='configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py',
    checkpoint='crpn_faster_rcnn_r50_caffe_fpn_1x_coco-c8283cca.pth',
    eval='bbox',
    metric=dict(bbox_mAP=40.4),
)
centripetalnet = dict(
    config='configs/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py',  # noqa
    checkpoint='centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804-3ccc61e5.pth',  # noqa
    eval='bbox',
    metric=dict(bbox_mAP=44.7),
)
cornernet = dict(
    config='configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py',
    checkpoint='cornernet_hourglass104_mstest_10x5_210e_coco_20200824_185720-5fefbf1c.pth',  # noqa
    eval='bbox',
    metric=dict(bbox_mAP=41.2),
)
dcn = dict(
    config='configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py',
    checkpoint='faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth',
    eval='bbox',
    metric=dict(bbox_mAP=41.3),
)
deformable_detr = dict(
    config='configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py',
    checkpoint='deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth',  # noqa
    eval='bbox',
    metric=dict(bbox_mAP=44.5),
)
detectors = dict(
    config='configs/detectors/detectors_htc_r50_1x_coco.py',
    checkpoint='detectors_htc_r50_1x_coco-329b1453.pth',
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=49.1, segm_mAP=42.6),
)
detr = dict(
    config='configs/detr/detr_r50_8x2_150e_coco.py',
    checkpoint='detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth',
    eval='bbox',
    metric=dict(bbox_mAP=40.1),
)
double_heads = dict(
    config='configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py',
    checkpoint='dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth',
    eval='bbox',
    metric=dict(bbox_mAP=40.0),
)
dynamic_rcnn = dict(
    config='configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py',
    checkpoint='dynamic_rcnn_r50_fpn_1x-62a3f276.pth',
    eval='bbox',
    metric=dict(bbox_mAP=38.9),
)
empirical_attention = dict(
    config='configs/empirical_attention/faster_rcnn_r50_fpn_attention_1111_1x_coco.py',  # noqa
    checkpoint='faster_rcnn_r50_fpn_attention_1111_1x_coco_20200130-403cccba.pth',  # noqa
    eval='bbox',
    metric=dict(bbox_mAP=40.0),
)
faster_rcnn = dict(
    config='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
    checkpoint='faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
    eval='bbox',
    metric=dict(bbox_mAP=37.4),
)
fcos = dict(
    config='configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py',  # noqa
    checkpoint='fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth',  # noqa
    eval='bbox',
    metric=dict(bbox_mAP=38.7),
)
foveabox = dict(
    config='configs/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py',
    checkpoint='fovea_align_r50_fpn_gn-head_4x4_2x_coco_20200203-8987880d.pth',
    eval='bbox',
    metric=dict(bbox_mAP=37.9),
)
free_anchor = dict(
    config='configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py',
    checkpoint='retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth',
    eval='bbox',
    metric=dict(bbox_mAP=38.7),
)
fsaf = dict(
    config='configs/fsaf/fsaf_r50_fpn_1x_coco.py',
    checkpoint='fsaf_r50_fpn_1x_coco-94ccc51f.pth',
    eval='bbox',
    metric=dict(bbox_mAP=37.4),
)
gcnet = dict(
    config='configs/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco.py',  # noqa
    checkpoint='mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200202-587b99aa.pth',  # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=40.4, segm_mAP=36.2),
)
gfl = dict(
    config='configs/gfl/gfl_r50_fpn_1x_coco.py',
    checkpoint='gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth',
    eval='bbox',
    metric=dict(bbox_mAP=40.2),
)
gn = dict(
    config='configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py',
    checkpoint='mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth',
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=40.1, segm_mAP=36.4),
)
gn_ws = dict(
    config='configs/gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py',
    checkpoint='faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth',
    eval='bbox',
    metric=dict(bbox_mAP=39.7),
)
grid_rcnn = dict(
    config='configs/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py',
    checkpoint='grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth',
    eval='bbox',
    metric=dict(bbox_mAP=40.4),
)
groie = dict(
    config='configs/groie/faster_rcnn_r50_fpn_groie_1x_coco.py',
    checkpoint='faster_rcnn_r50_fpn_groie_1x_coco_20200604_211715-66ee9516.pth',  # noqa
    eval='bbox',
    metric=dict(bbox_mAP=38.3),
)
guided_anchoring = [
    dict(
        config='configs/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco.py',  # noqa
        checkpoint='ga_retinanet_r50_caffe_fpn_1x_coco_20201020-39581c6f.pth',
        eval='bbox',
        metric=dict(bbox_mAP=36.9),
    ),
    dict(
        config='configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py',
        checkpoint='ga_faster_r50_caffe_fpn_1x_coco_20200702_000718-a11ccfe6.pth',  # noqa
        eval='bbox',
        metric=dict(bbox_mAP=39.6),
    ),
]
hrnet = dict(
    config='configs/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco.py',
    checkpoint='faster_rcnn_hrnetv2p_w18_1x_coco_20200130-56651a6d.pth',
    eval='bbox',
    metric=dict(bbox_mAP=36.9),
)
htc = dict(
    config='configs/htc/htc_r50_fpn_1x_coco.py',
    checkpoint='htc_r50_fpn_1x_coco_20200317-7332cf16.pth',
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=42.3, segm_mAP=37.4),
)
libra_rcnn = dict(
    config='configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py',
    checkpoint='libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth',
    eval='bbox',
    metric=dict(bbox_mAP=38.3),
)
mask_rcnn = dict(
    config='configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py',
    checkpoint='mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth',
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=38.2, segm_mAP=34.7),
)
ms_rcnn = dict(
    config='configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py',
    checkpoint='ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth',
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=38.2, segm_mAP=36.0),
)
nas_fcos = dict(
    config='configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py',  # noqa
    checkpoint='nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200520-1bdba3ce.pth',  # noqa
    eval='bbox',
    metric=dict(bbox_mAP=39.4),
)
nas_fpn = dict(
    config='configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py',
    checkpoint='retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth',
    eval='bbox',
    metric=dict(bbox_mAP=40.5),
)
paa = dict(
    config='configs/paa/paa_r50_fpn_1x_coco.py',
    checkpoint='paa_r50_fpn_1x_coco_20200821-936edec3.pth',
    eval='bbox',
    metric=dict(bbox_mAP=40.4),
)
pafpn = dict(
    config='configs/pafpn/faster_rcnn_r50_pafpn_1x_coco.py',
    checkpoint='faster_rcnn_r50_pafpn_1x_coco_bbox_mAP-0.375_20200503_105836-b7b4b9bd.pth',  # noqa
    eval='bbox',
    metric=dict(bbox_mAP=37.5),
)
pisa = dict(
    config='configs/pisa/pisa_faster_rcnn_r50_fpn_1x_coco.py',
    checkpoint='pisa_faster_rcnn_r50_fpn_1x_coco-dea93523.pth',
    eval='bbox',
    metric=dict(bbox_mAP=38.4),
)
point_rend = dict(
    config='configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py',
    checkpoint='point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth',
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=38.4, segm_mAP=36.3),
)
regnet = dict(
    config='configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py',
    checkpoint='mask_rcnn_regnetx-3.2GF_fpn_1x_coco_20200520_163141-2a9d1814.pth',  # noqa
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=40.4, segm_mAP=36.7),
)
reppoints = dict(
    config='configs/reppoints/reppoints_moment_r50_fpn_1x_coco.py',
    checkpoint='reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth',
    eval='bbox',
    metric=dict(bbox_mAP=37.0),
)
res2net = dict(
    config='configs/res2net/faster_rcnn_r2_101_fpn_2x_coco.py',
    checkpoint='faster_rcnn_r2_101_fpn_2x_coco-175f1da6.pth',
    eval='bbox',
    metric=dict(bbox_mAP=43.0),
)
resnest = dict(
    config='configs/resnest/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py',  # noqa
    checkpoint='faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco_20200926_125502-20289c16.pth',  # noqa
    eval='bbox',
    metric=dict(bbox_mAP=42.0),
)
retinanet = dict(
    config='configs/retinanet/retinanet_r50_fpn_1x_coco.py',
    checkpoint='retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth',
    eval='bbox',
    metric=dict(bbox_mAP=36.5),
)
rpn = dict(
    config='configs/rpn/rpn_r50_fpn_1x_coco.py',
    checkpoint='rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth',
    eval='proposal_fast',
    metric=dict(AR_1000=58.2),
)
sabl = [
    dict(
        config='configs/sabl/sabl_retinanet_r50_fpn_1x_coco.py ',
        checkpoint='sabl_retinanet_r50_fpn_1x_coco-6c54fd4f.pth',
        eval='bbox',
        metric=dict(bbox_mAP=37.7),
    ),
    dict(
        config='configs/sabl/sabl_faster_rcnn_r50_fpn_1x_coco.py',
        checkpoint='sabl_faster_rcnn_r50_fpn_1x_coco-e867595b.pth',
        eval='bbox',
        metric=dict(bbox_mAP=39.9),
    ),
]
scnet = dict(
    config='configs/scnet/scnet_r50_fpn_1x_coco.py',
    checkpoint='scnet_r50_fpn_1x_coco-c3f09857.pth',
    eval='bbox',
    metric=dict(bbox_mAP=43.5),
)
sparse_rcnn = dict(
    config='configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py',
    checkpoint='sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth',
    eval='bbox',
    metric=dict(bbox_mAP=37.9),
)
ssd = dict(
    config='configs/ssd/ssd300_coco.py',
    checkpoint='ssd300_coco_20200307-a92d2092.pth',
    eval='bbox',
    metric=dict(bbox_mAP=25.6),
)
tridentnet = dict(
    config='configs/tridentnet/tridentnet_r50_caffe_1x_coco.py',
    checkpoint='tridentnet_r50_caffe_1x_coco_20201230_141838-2ec0b530.pth',
    eval='bbox',
    metric=dict(bbox_mAP=37.6),
)
vfnet = dict(
    config='configs/vfnet/vfnet_r50_fpn_1x_coco.py',
    checkpoint='vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth',
    eval='bbox',
    metric=dict(bbox_mAP=41.6),
)
yolact = dict(
    config='configs/yolact/yolact_r50_1x8_coco.py',
    checkpoint='yolact_r50_1x8_coco_20200908-f38d58df.pth',
    eval=['bbox', 'segm'],
    metric=dict(bbox_mAP=31.2, segm_mAP=29.0),
)
yolo = dict(
    config='configs/yolo/yolov3_d53_320_273e_coco.py',
    checkpoint='yolov3_d53_320_273e_coco-421362b6.pth',
    eval='bbox',
    metric=dict(bbox_mAP=27.9),
)
yolof = dict(
    config='configs/yolof/yolof_r50_c5_8x8_1x_coco.py',
    checkpoint='yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth',
    eval='bbox',
    metric=dict(bbox_mAP=37.5),
)
centernet = dict(
    config='configs/centernet/centernet_resnet18_dcnv2_140e_coco.py',
    checkpoint='centernet_resnet18_dcnv2_140e_coco_20210520_101209-da388ba2.pth',  # noqa
    eval='bbox',
    metric=dict(bbox_mAP=29.5),
)
# yapf: enable
