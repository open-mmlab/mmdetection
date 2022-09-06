PARTITION=$1
CHECKPOINT_DIR=$2
WORK_DIR=$3
CPUS_PER_TASK=${4:-2}

echo 'configs/atss/atss_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION atss_r50_fpn_1x_coco configs/atss/atss_r50_fpn_1x_coco.py $CHECKPOINT_DIR/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth --work-dir $WORK_DIR/atss_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29666  &
echo 'configs/autoassign/autoassign_r50-caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION autoassign_r50-caffe_fpn_1x_coco configs/autoassign/autoassign_r50-caffe_fpn_1x_coco.py $CHECKPOINT_DIR/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth --work-dir $WORK_DIR/autoassign_r50-caffe_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29667  &
echo 'configs/carafe/faster-rcnn_r50_fpn-carafe_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION faster-rcnn_r50_fpn-carafe_1x_coco configs/carafe/faster-rcnn_r50_fpn-carafe_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.386_20200504_175733-385a75b7.pth --work-dir $WORK_DIR/faster-rcnn_r50_fpn-carafe_1x_coco --cfg-option env_cfg.dist_cfg.port=29668  &
echo 'configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION cascade-rcnn_r50_fpn_1x_coco configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth --work-dir $WORK_DIR/cascade-rcnn_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29669  &
echo 'configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION cascade-mask-rcnn_r50_fpn_1x_coco configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth --work-dir $WORK_DIR/cascade-mask-rcnn_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29670  &
echo 'configs/cascade_rpn/cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco configs/cascade_rpn/cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco.py $CHECKPOINT_DIR/crpn_faster_rcnn_r50_caffe_fpn_1x_coco-c8283cca.pth --work-dir $WORK_DIR/cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29671  &
echo 'configs/centernet/centernet_r18-dcnv2_8xb16-crop512-140e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION centernet_r18-dcnv2_8xb16-crop512-140e_coco configs/centernet/centernet_r18-dcnv2_8xb16-crop512-140e_coco.py $CHECKPOINT_DIR/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth --work-dir $WORK_DIR/centernet_r18-dcnv2_8xb16-crop512-140e_coco --cfg-option env_cfg.dist_cfg.port=29672  &
echo 'configs/centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco configs/centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py $CHECKPOINT_DIR/centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804-3ccc61e5.pth --work-dir $WORK_DIR/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco --cfg-option env_cfg.dist_cfg.port=29673  &
echo 'configs/convnext/cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco configs/convnext/cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py $CHECKPOINT_DIR/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220510_201004-3d24f5a4.pth --work-dir $WORK_DIR/cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco --cfg-option env_cfg.dist_cfg.port=29674  &
echo 'configs/cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION cornernet_hourglass104_8xb6-210e-mstest_coco configs/cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py $CHECKPOINT_DIR/cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618-79b44c30.pth --work-dir $WORK_DIR/cornernet_hourglass104_8xb6-210e-mstest_coco --cfg-option env_cfg.dist_cfg.port=29675  &
echo 'configs/dcn/faster-rcnn_r50-dconv-c3-c5_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION faster-rcnn_r50-dconv-c3-c5_fpn_1x_coco configs/dcn/faster-rcnn_r50-dconv-c3-c5_fpn_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth --work-dir $WORK_DIR/faster-rcnn_r50-dconv-c3-c5_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29676  &
echo 'configs/dcnv2/faster-rcnn_r50_fpn_mdpool_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION faster-rcnn_r50_fpn_mdpool_1x_coco configs/dcnv2/faster-rcnn_r50_fpn_mdpool_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_fpn_mdpool_1x_coco_20200307-c0df27ff.pth --work-dir $WORK_DIR/faster-rcnn_r50_fpn_mdpool_1x_coco --cfg-option env_cfg.dist_cfg.port=29677  &
echo 'configs/ddod/ddod_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION ddod_r50_fpn_1x_coco configs/ddod/ddod_r50_fpn_1x_coco.py $CHECKPOINT_DIR/ddod_r50_fpn_1x_coco_20220523_223737-29b2fc67.pth --work-dir $WORK_DIR/ddod_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29678  &
echo 'configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION deformable-detr_r50_16xb2-50e_coco configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py $CHECKPOINT_DIR/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth --work-dir $WORK_DIR/deformable-detr_r50_16xb2-50e_coco --cfg-option env_cfg.dist_cfg.port=29679  &
echo 'configs/detectors/detectors_htc-r50_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION detectors_htc-r50_1x_coco configs/detectors/detectors_htc-r50_1x_coco.py $CHECKPOINT_DIR/detectors_htc_r50_1x_coco-329b1453.pth --work-dir $WORK_DIR/detectors_htc-r50_1x_coco --cfg-option env_cfg.dist_cfg.port=29680  &
echo 'configs/detr/detr_r50_8xb2-150e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION detr_r50_8xb2-150e_coco configs/detr/detr_r50_8xb2-150e_coco.py $CHECKPOINT_DIR/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth --work-dir $WORK_DIR/detr_r50_8xb2-150e_coco --cfg-option env_cfg.dist_cfg.port=29681  &
echo 'configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION dh-faster-rcnn_r50_fpn_1x_coco configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth --work-dir $WORK_DIR/dh-faster-rcnn_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29682  &
echo 'configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION atss_r50_fpn_dyhead_1x_coco configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py $CHECKPOINT_DIR/atss_r50_fpn_dyhead_4x4_1x_coco_20211219_023314-eaa620c6.pth --work-dir $WORK_DIR/atss_r50_fpn_dyhead_1x_coco --cfg-option env_cfg.dist_cfg.port=29683  &
echo 'configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION dynamic-rcnn_r50_fpn_1x_coco configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/dynamic_rcnn_r50_fpn_1x-62a3f276.pth --work-dir $WORK_DIR/dynamic-rcnn_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29684  &
echo 'configs/efficientnet/retinanet_effb3_fpn_8xb4-crop896-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION retinanet_effb3_fpn_8xb4-crop896-1x_coco configs/efficientnet/retinanet_effb3_fpn_8xb4-crop896-1x_coco.py $CHECKPOINT_DIR/retinanet_effb3_fpn_crop896_8x4_1x_coco_20220322_234806-615a0dda.pth --work-dir $WORK_DIR/retinanet_effb3_fpn_8xb4-crop896-1x_coco --cfg-option env_cfg.dist_cfg.port=29685  &
echo 'configs/empirical_attention/faster-rcnn_r50-attn1111_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION faster-rcnn_r50-attn1111_fpn_1x_coco configs/empirical_attention/faster-rcnn_r50-attn1111_fpn_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_fpn_attention_1111_1x_coco_20200130-403cccba.pth --work-dir $WORK_DIR/faster-rcnn_r50-attn1111_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29686  &
echo 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION faster-rcnn_r50_fpn_1x_coco configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --work-dir $WORK_DIR/faster-rcnn_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29687  &
echo 'configs/fcos/fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco configs/fcos/fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py $CHECKPOINT_DIR/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth --work-dir $WORK_DIR/fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco --cfg-option env_cfg.dist_cfg.port=29688  &
echo 'configs/foveabox/fovea_r50_fpn_gn-head-align_4xb4-2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION fovea_r50_fpn_gn-head-align_4xb4-2x_coco configs/foveabox/fovea_r50_fpn_gn-head-align_4xb4-2x_coco.py $CHECKPOINT_DIR/fovea_align_r50_fpn_gn-head_4x4_2x_coco_20200203-8987880d.pth --work-dir $WORK_DIR/fovea_r50_fpn_gn-head-align_4xb4-2x_coco --cfg-option env_cfg.dist_cfg.port=29689  &
echo 'configs/fpg/mask-rcnn_r50_fpg_crop640-50e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION mask-rcnn_r50_fpg_crop640-50e_coco configs/fpg/mask-rcnn_r50_fpg_crop640-50e_coco.py $CHECKPOINT_DIR/mask_rcnn_r50_fpg_crop640_50e_coco_20220311_011857-233b8334.pth --work-dir $WORK_DIR/mask-rcnn_r50_fpg_crop640-50e_coco --cfg-option env_cfg.dist_cfg.port=29690  &
echo 'configs/free_anchor/freeanchor_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION freeanchor_r50_fpn_1x_coco configs/free_anchor/freeanchor_r50_fpn_1x_coco.py $CHECKPOINT_DIR/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth --work-dir $WORK_DIR/freeanchor_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29691  &
echo 'configs/fsaf/fsaf_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION fsaf_r50_fpn_1x_coco configs/fsaf/fsaf_r50_fpn_1x_coco.py $CHECKPOINT_DIR/fsaf_r50_fpn_1x_coco-94ccc51f.pth --work-dir $WORK_DIR/fsaf_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29692  &
echo 'configs/gcnet/mask-rcnn_r50-syncbn-gcb-r16-c3-c5_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION mask-rcnn_r50-syncbn-gcb-r16-c3-c5_fpn_1x_coco configs/gcnet/mask-rcnn_r50-syncbn-gcb-r16-c3-c5_fpn_1x_coco.py $CHECKPOINT_DIR/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200202-587b99aa.pth --work-dir $WORK_DIR/mask-rcnn_r50-syncbn-gcb-r16-c3-c5_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29693  &
echo 'configs/gfl/gfl_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION gfl_r50_fpn_1x_coco configs/gfl/gfl_r50_fpn_1x_coco.py $CHECKPOINT_DIR/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth --work-dir $WORK_DIR/gfl_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29694  &
echo 'configs/ghm/retinanet_r50_fpn_ghm-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION retinanet_r50_fpn_ghm-1x_coco configs/ghm/retinanet_r50_fpn_ghm-1x_coco.py $CHECKPOINT_DIR/retinanet_ghm_r50_fpn_1x_coco_20200130-a437fda3.pth --work-dir $WORK_DIR/retinanet_r50_fpn_ghm-1x_coco --cfg-option env_cfg.dist_cfg.port=29695  &
echo 'configs/gn/mask-rcnn_r50_fpn_gn-all_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION mask-rcnn_r50_fpn_gn-all_2x_coco configs/gn/mask-rcnn_r50_fpn_gn-all_2x_coco.py $CHECKPOINT_DIR/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth --work-dir $WORK_DIR/mask-rcnn_r50_fpn_gn-all_2x_coco --cfg-option env_cfg.dist_cfg.port=29696  &
echo 'configs/gn+ws/faster-rcnn_r50_fpn_gn-ws-all_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION faster-rcnn_r50_fpn_gn-ws-all_1x_coco configs/gn+ws/faster-rcnn_r50_fpn_gn-ws-all_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth --work-dir $WORK_DIR/faster-rcnn_r50_fpn_gn-ws-all_1x_coco --cfg-option env_cfg.dist_cfg.port=29697  &
echo 'configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION grid-rcnn_r50_fpn_gn-head_2x_coco configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_coco.py $CHECKPOINT_DIR/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth --work-dir $WORK_DIR/grid-rcnn_r50_fpn_gn-head_2x_coco --cfg-option env_cfg.dist_cfg.port=29698  &
echo 'configs/groie/faste-rcnn_r50_fpn_groie_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION faste-rcnn_r50_fpn_groie_1x_coco configs/groie/faste-rcnn_r50_fpn_groie_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_fpn_groie_1x_coco_20200604_211715-66ee9516.pth --work-dir $WORK_DIR/faste-rcnn_r50_fpn_groie_1x_coco --cfg-option env_cfg.dist_cfg.port=29699  &
echo 'configs/guided_anchoring/ga-retinanet_r50-caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION ga-retinanet_r50-caffe_fpn_1x_coco configs/guided_anchoring/ga-retinanet_r50-caffe_fpn_1x_coco.py $CHECKPOINT_DIR/ga_retinanet_r50_caffe_fpn_1x_coco_20201020-39581c6f.pth --work-dir $WORK_DIR/ga-retinanet_r50-caffe_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29700  &
echo 'configs/hrnet/faster-rcnn_hrnetv2p-w18-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION faster-rcnn_hrnetv2p-w18-1x_coco configs/hrnet/faster-rcnn_hrnetv2p-w18-1x_coco.py $CHECKPOINT_DIR/faster_rcnn_hrnetv2p_w18_1x_coco_20200130-56651a6d.pth --work-dir $WORK_DIR/faster-rcnn_hrnetv2p-w18-1x_coco --cfg-option env_cfg.dist_cfg.port=29701  &
echo 'configs/htc/htc_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION htc_r50_fpn_1x_coco configs/htc/htc_r50_fpn_1x_coco.py $CHECKPOINT_DIR/htc_r50_fpn_1x_coco_20200317-7332cf16.pth --work-dir $WORK_DIR/htc_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29702  &
echo 'configs/instaboost/mask-rcnn_r50_fpn_instaboost-4x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION mask-rcnn_r50_fpn_instaboost-4x_coco configs/instaboost/mask-rcnn_r50_fpn_instaboost-4x_coco.py $CHECKPOINT_DIR/mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-d025f83a.pth --work-dir $WORK_DIR/mask-rcnn_r50_fpn_instaboost-4x_coco --cfg-option env_cfg.dist_cfg.port=29703  &
echo 'configs/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION libra-faster-rcnn_r50_fpn_1x_coco configs/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth --work-dir $WORK_DIR/libra-faster-rcnn_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29704  &
echo 'configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION mask2former_r50_8xb2-lsj-50e_coco-panoptic configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py $CHECKPOINT_DIR/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth --work-dir $WORK_DIR/mask2former_r50_8xb2-lsj-50e_coco-panoptic --cfg-option env_cfg.dist_cfg.port=29705  &
echo 'configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION mask-rcnn_r50_fpn_1x_coco configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth --work-dir $WORK_DIR/mask-rcnn_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29706  &
echo 'configs/maskformer/maskformer_r50_ms-16xb1-75e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION maskformer_r50_ms-16xb1-75e_coco configs/maskformer/maskformer_r50_ms-16xb1-75e_coco.py $CHECKPOINT_DIR/maskformer_r50_mstrain_16x1_75e_coco_20220221_141956-bc2699cb.pth --work-dir $WORK_DIR/maskformer_r50_ms-16xb1-75e_coco --cfg-option env_cfg.dist_cfg.port=29707  &
echo 'configs/ms_rcnn/ms-rcnn_r50-caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION ms-rcnn_r50-caffe_fpn_1x_coco configs/ms_rcnn/ms-rcnn_r50-caffe_fpn_1x_coco.py $CHECKPOINT_DIR/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth --work-dir $WORK_DIR/ms-rcnn_r50-caffe_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29708  &
echo 'configs/nas_fcos/nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco configs/nas_fcos/nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco.py $CHECKPOINT_DIR/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200520-1bdba3ce.pth --work-dir $WORK_DIR/nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco --cfg-option env_cfg.dist_cfg.port=29709  &
echo 'configs/nas_fpn/retinanet_r50_nasfpn_crop640-50e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION retinanet_r50_nasfpn_crop640-50e_coco configs/nas_fpn/retinanet_r50_nasfpn_crop640-50e_coco.py $CHECKPOINT_DIR/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth --work-dir $WORK_DIR/retinanet_r50_nasfpn_crop640-50e_coco --cfg-option env_cfg.dist_cfg.port=29710  &
echo 'configs/paa/paa_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION paa_r50_fpn_1x_coco configs/paa/paa_r50_fpn_1x_coco.py $CHECKPOINT_DIR/paa_r50_fpn_1x_coco_20200821-936edec3.pth --work-dir $WORK_DIR/paa_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29711  &
echo 'configs/pafpn/faster-rcnn_r50_pafpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION faster-rcnn_r50_pafpn_1x_coco configs/pafpn/faster-rcnn_r50_pafpn_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_pafpn_1x_coco_bbox_mAP-0.375_20200503_105836-b7b4b9bd.pth --work-dir $WORK_DIR/faster-rcnn_r50_pafpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29712  &
echo 'configs/panoptic_fpn/panoptic-fpn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION panoptic-fpn_r50_fpn_1x_coco configs/panoptic_fpn/panoptic-fpn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/panoptic_fpn_r50_fpn_1x_coco_20210821_101153-9668fd13.pth --work-dir $WORK_DIR/panoptic-fpn_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29713  &
echo 'configs/pisa/faster-rcnn_r50_fpn_pisa_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION faster-rcnn_r50_fpn_pisa_1x_coco configs/pisa/faster-rcnn_r50_fpn_pisa_1x_coco.py $CHECKPOINT_DIR/pisa_faster_rcnn_r50_fpn_1x_coco-dea93523.pth --work-dir $WORK_DIR/faster-rcnn_r50_fpn_pisa_1x_coco --cfg-option env_cfg.dist_cfg.port=29714  &
echo 'configs/point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION point-rend_r50-caffe_fpn_ms-1x_coco configs/point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py $CHECKPOINT_DIR/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth --work-dir $WORK_DIR/point-rend_r50-caffe_fpn_ms-1x_coco --cfg-option env_cfg.dist_cfg.port=29715  &
echo 'configs/pvt/retinanet_pvt-s_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION retinanet_pvt-s_fpn_1x_coco configs/pvt/retinanet_pvt-s_fpn_1x_coco.py $CHECKPOINT_DIR/retinanet_pvt-s_fpn_1x_coco_20210906_142921-b6c94a5b.pth --work-dir $WORK_DIR/retinanet_pvt-s_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29716  &
echo 'configs/queryinst/queryinst_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION queryinst_r50_fpn_1x_coco configs/queryinst/queryinst_r50_fpn_1x_coco.py $CHECKPOINT_DIR/queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth --work-dir $WORK_DIR/queryinst_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29717  &
echo 'configs/regnet/mask-rcnn_regnetx-3.2GF_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION mask-rcnn_regnetx-3.2GF_fpn_1x_coco configs/regnet/mask-rcnn_regnetx-3.2GF_fpn_1x_coco.py $CHECKPOINT_DIR/mask_rcnn_regnetx-3.2GF_fpn_1x_coco_20200520_163141-2a9d1814.pth --work-dir $WORK_DIR/mask-rcnn_regnetx-3.2GF_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29718  &
echo 'configs/reppoints/reppoints-moment_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION reppoints-moment_r50_fpn_1x_coco configs/reppoints/reppoints-moment_r50_fpn_1x_coco.py $CHECKPOINT_DIR/reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth --work-dir $WORK_DIR/reppoints-moment_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29719  &
echo 'configs/res2net/faster-rcnn_res2net-101_fpn_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION faster-rcnn_res2net-101_fpn_2x_coco configs/res2net/faster-rcnn_res2net-101_fpn_2x_coco.py $CHECKPOINT_DIR/faster_rcnn_r2_101_fpn_2x_coco-175f1da6.pth --work-dir $WORK_DIR/faster-rcnn_res2net-101_fpn_2x_coco --cfg-option env_cfg.dist_cfg.port=29720  &
echo 'configs/resnest/faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco configs/resnest/faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco.py $CHECKPOINT_DIR/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco_20200926_125502-20289c16.pth --work-dir $WORK_DIR/faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco --cfg-option env_cfg.dist_cfg.port=29721  &
echo 'configs/resnet_strikes_back/mask-rcnn_r50-rsb-pre_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION mask-rcnn_r50-rsb-pre_fpn_1x_coco configs/resnet_strikes_back/mask-rcnn_r50-rsb-pre_fpn_1x_coco.py $CHECKPOINT_DIR/mask_rcnn_r50_fpn_rsb-pretrain_1x_coco_20220113_174054-06ce8ba0.pth --work-dir $WORK_DIR/mask-rcnn_r50-rsb-pre_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29722  &
echo 'configs/retinanet/retinanet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION retinanet_r50_fpn_1x_coco configs/retinanet/retinanet_r50_fpn_1x_coco.py $CHECKPOINT_DIR/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --work-dir $WORK_DIR/retinanet_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29723  &
echo 'configs/rpn/rpn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION rpn_r50_fpn_1x_coco configs/rpn/rpn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth --work-dir $WORK_DIR/rpn_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29724  &
echo 'configs/sabl/sabl-retinanet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION sabl-retinanet_r50_fpn_1x_coco configs/sabl/sabl-retinanet_r50_fpn_1x_coco.py $CHECKPOINT_DIR/sabl_retinanet_r50_fpn_1x_coco-6c54fd4f.pth --work-dir $WORK_DIR/sabl-retinanet_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29725  &
echo 'configs/sabl/sabl-faster-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION sabl-faster-rcnn_r50_fpn_1x_coco configs/sabl/sabl-faster-rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/sabl_faster_rcnn_r50_fpn_1x_coco-e867595b.pth --work-dir $WORK_DIR/sabl-faster-rcnn_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29726  &
echo 'configs/scnet/scnet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION scnet_r50_fpn_1x_coco configs/scnet/scnet_r50_fpn_1x_coco.py $CHECKPOINT_DIR/scnet_r50_fpn_1x_coco-c3f09857.pth --work-dir $WORK_DIR/scnet_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29727  &
echo 'configs/scratch/mask-rcnn_r50-scratch_fpn_gn-all_6x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION mask-rcnn_r50-scratch_fpn_gn-all_6x_coco configs/scratch/mask-rcnn_r50-scratch_fpn_gn-all_6x_coco.py $CHECKPOINT_DIR/scratch_mask_rcnn_r50_fpn_gn_6x_bbox_mAP-0.412__segm_mAP-0.374_20200201_193051-1e190a40.pth --work-dir $WORK_DIR/mask-rcnn_r50-scratch_fpn_gn-all_6x_coco --cfg-option env_cfg.dist_cfg.port=29728  &
echo 'configs/solo/decoupled-solo_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION decoupled-solo_r50_fpn_1x_coco configs/solo/decoupled-solo_r50_fpn_1x_coco.py $CHECKPOINT_DIR/decoupled_solo_r50_fpn_1x_coco_20210820_233348-6337c589.pth --work-dir $WORK_DIR/decoupled-solo_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29729  &
echo 'configs/solov2/solov2_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION solov2_r50_fpn_1x_coco configs/solov2/solov2_r50_fpn_1x_coco.py $CHECKPOINT_DIR/solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth --work-dir $WORK_DIR/solov2_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29730  &
echo 'configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION sparse-rcnn_r50_fpn_1x_coco configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth --work-dir $WORK_DIR/sparse-rcnn_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29731  &
echo 'configs/ssd/ssd300_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION ssd300_coco configs/ssd/ssd300_coco.py $CHECKPOINT_DIR/ssd300_coco_20210803_015428-d231a06e.pth --work-dir $WORK_DIR/ssd300_coco --cfg-option env_cfg.dist_cfg.port=29732  &
echo 'configs/ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION ssdlite_mobilenetv2-scratch_8xb24-600e_coco configs/ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py $CHECKPOINT_DIR/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth --work-dir $WORK_DIR/ssdlite_mobilenetv2-scratch_8xb24-600e_coco --cfg-option env_cfg.dist_cfg.port=29733  &
echo 'configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION mask-rcnn_swin-t-p4-w7_fpn_1x_coco configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py $CHECKPOINT_DIR/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth --work-dir $WORK_DIR/mask-rcnn_swin-t-p4-w7_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29734  &
echo 'configs/tood/tood_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION tood_r50_fpn_1x_coco configs/tood/tood_r50_fpn_1x_coco.py $CHECKPOINT_DIR/tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth --work-dir $WORK_DIR/tood_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29735  &
echo 'configs/tridentnet/tridentnet_r50-caffe_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION tridentnet_r50-caffe_1x_coco configs/tridentnet/tridentnet_r50-caffe_1x_coco.py $CHECKPOINT_DIR/tridentnet_r50_caffe_1x_coco_20201230_141838-2ec0b530.pth --work-dir $WORK_DIR/tridentnet_r50-caffe_1x_coco --cfg-option env_cfg.dist_cfg.port=29736  &
echo 'configs/vfnet/vfnet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION vfnet_r50_fpn_1x_coco configs/vfnet/vfnet_r50_fpn_1x_coco.py $CHECKPOINT_DIR/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth --work-dir $WORK_DIR/vfnet_r50_fpn_1x_coco --cfg-option env_cfg.dist_cfg.port=29737  &
echo 'configs/yolact/yolact_r50_1xb8-55e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION yolact_r50_1xb8-55e_coco configs/yolact/yolact_r50_1xb8-55e_coco.py $CHECKPOINT_DIR/yolact_r50_1x8_coco_20200908-f38d58df.pth --work-dir $WORK_DIR/yolact_r50_1xb8-55e_coco --cfg-option env_cfg.dist_cfg.port=29738  &
echo 'configs/yolo/yolov3_d53_8xb8-320-273e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION yolov3_d53_8xb8-320-273e_coco configs/yolo/yolov3_d53_8xb8-320-273e_coco.py $CHECKPOINT_DIR/yolov3_d53_320_273e_coco-421362b6.pth --work-dir $WORK_DIR/yolov3_d53_8xb8-320-273e_coco --cfg-option env_cfg.dist_cfg.port=29739  &
echo 'configs/yolof/yolof_r50-c5_8xb8-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION yolof_r50-c5_8xb8-1x_coco configs/yolof/yolof_r50-c5_8xb8-1x_coco.py $CHECKPOINT_DIR/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth --work-dir $WORK_DIR/yolof_r50-c5_8xb8-1x_coco --cfg-option env_cfg.dist_cfg.port=29740  &
echo 'configs/yolox/yolox_tiny_8xb8-300e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK tools/slurm_test.sh $PARTITION yolox_tiny_8xb8-300e_coco configs/yolox/yolox_tiny_8xb8-300e_coco.py $CHECKPOINT_DIR/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth --work-dir $WORK_DIR/yolox_tiny_8xb8-300e_coco --cfg-option env_cfg.dist_cfg.port=29741  &
