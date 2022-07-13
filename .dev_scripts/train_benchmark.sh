PARTITION=$1
WORK_DIR=$2
CPUS_PER_TASK=${3:-4}

echo 'configs/albu_example/mask_rcnn_r50_fpn_albu_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask_rcnn_r50_fpn_albu_1x_coco configs/albu_example/mask_rcnn_r50_fpn_albu_1x_coco.py $WORK_DIR/mask_rcnn_r50_fpn_albu_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/atss/atss_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION atss_r50_fpn_1x_coco configs/atss/atss_r50_fpn_1x_coco.py $WORK_DIR/atss_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION autoassign_r50_fpn_8x2_1x_coco configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py $WORK_DIR/autoassign_r50_fpn_8x2_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/carafe/faster_rcnn_r50_fpn_carafe_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster_rcnn_r50_fpn_carafe_1x_coco configs/carafe/faster_rcnn_r50_fpn_carafe_1x_coco.py $WORK_DIR/faster_rcnn_r50_fpn_carafe_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION cascade_rcnn_r50_fpn_1x_coco configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py $WORK_DIR/cascade_rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION cascade_mask_rcnn_r50_fpn_1x_coco configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py $WORK_DIR/cascade_mask_rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION crpn_faster_rcnn_r50_caffe_fpn_1x_coco configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py $WORK_DIR/crpn_faster_rcnn_r50_caffe_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/centernet/centernet_resnet18_dcnv2_140e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION centernet_resnet18_dcnv2_140e_coco configs/centernet/centernet_resnet18_dcnv2_140e_coco.py $WORK_DIR/centernet_resnet18_dcnv2_140e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/centernet/centernet_update_r50_caffe_fpn_mstrain_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION centernet_update_r50_caffe_fpn_mstrain_1x_coco configs/centernet/centernet_update_r50_caffe_fpn_mstrain_1x_coco.py $WORK_DIR/centernet_update_r50_caffe_fpn_mstrain_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py' &
GPUS=16  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION centripetalnet_hourglass104_mstest_16x6_210e_coco configs/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py $WORK_DIR/centripetalnet_hourglass104_mstest_16x6_210e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION cornernet_hourglass104_mstest_8x6_210e_coco configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py $WORK_DIR/cornernet_hourglass104_mstest_8x6_210e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/convnext/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco configs/convnext/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco.py $WORK_DIR/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py $WORK_DIR/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/dcnv2/faster_rcnn_r50_fpn_mdpool_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster_rcnn_r50_fpn_mdpool_1x_coco configs/dcnv2/faster_rcnn_r50_fpn_mdpool_1x_coco.py $WORK_DIR/faster_rcnn_r50_fpn_mdpool_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/ddod/ddod_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION ddod_r50_fpn_1x_coco configs/ddod/ddod_r50_fpn_1x_coco.py $WORK_DIR/ddod_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/detectors/detectors_htc_r50_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION detectors_htc_r50_1x_coco configs/detectors/detectors_htc_r50_1x_coco.py $WORK_DIR/detectors_htc_r50_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py' &
GPUS=16  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION deformable_detr_r50_16x2_50e_coco configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py $WORK_DIR/deformable_detr_r50_16x2_50e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/detr/detr_r50_8x2_150e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION detr_r50_8x2_150e_coco configs/detr/detr_r50_8x2_150e_coco.py $WORK_DIR/detr_r50_8x2_150e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION dh_faster_rcnn_r50_fpn_1x_coco configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py $WORK_DIR/dh_faster_rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION dynamic_rcnn_r50_fpn_1x_coco configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py $WORK_DIR/dynamic_rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION atss_r50_fpn_dyhead_1x_coco configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py $WORK_DIR/atss_r50_fpn_dyhead_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/efficientnet/retinanet_effb3_fpn_crop896_8x4_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_effb3_fpn_crop896_8x4_1x_coco configs/efficientnet/retinanet_effb3_fpn_crop896_8x4_1x_coco.py $WORK_DIR/retinanet_effb3_fpn_crop896_8x4_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/empirical_attention/faster_rcnn_r50_fpn_attention_0010_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster_rcnn_r50_fpn_attention_0010_1x_coco configs/empirical_attention/faster_rcnn_r50_fpn_attention_0010_1x_coco.py $WORK_DIR/faster_rcnn_r50_fpn_attention_0010_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster_rcnn_r50_fpn_1x_coco configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py $WORK_DIR/faster_rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster_rcnn_r50_caffe_dc5_mstrain_1x_coco configs/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco.py $WORK_DIR/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster_rcnn_r50_fpn_ohem_1x_coco configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py $WORK_DIR/faster_rcnn_r50_fpn_ohem_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py $WORK_DIR/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION fovea_align_r50_fpn_gn-head_4x4_2x_coco configs/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py $WORK_DIR/fovea_align_r50_fpn_gn-head_4x4_2x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/fpg/retinanet_r50_fpg_crop640_50e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_r50_fpg_crop640_50e_coco configs/fpg/retinanet_r50_fpg_crop640_50e_coco.py $WORK_DIR/retinanet_r50_fpg_crop640_50e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_free_anchor_r50_fpn_1x_coco configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py $WORK_DIR/retinanet_free_anchor_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/fsaf/fsaf_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION fsaf_r50_fpn_1x_coco configs/fsaf/fsaf_r50_fpn_1x_coco.py $WORK_DIR/fsaf_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py $WORK_DIR/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/gfl/gfl_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION gfl_r50_fpn_1x_coco configs/gfl/gfl_r50_fpn_1x_coco.py $WORK_DIR/gfl_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/ghm/retinanet_ghm_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_ghm_r50_fpn_1x_coco configs/ghm/retinanet_ghm_r50_fpn_1x_coco.py $WORK_DIR/retinanet_ghm_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask_rcnn_r50_fpn_gn-all_2x_coco configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py $WORK_DIR/mask_rcnn_r50_fpn_gn-all_2x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster_rcnn_r50_fpn_gn_ws-all_1x_coco configs/gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py $WORK_DIR/faster_rcnn_r50_fpn_gn_ws-all_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION grid_rcnn_r50_fpn_gn-head_2x_coco configs/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py $WORK_DIR/grid_rcnn_r50_fpn_gn-head_2x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/groie/mask_rcnn_r50_fpn_groie_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask_rcnn_r50_fpn_groie_1x_coco configs/groie/mask_rcnn_r50_fpn_groie_1x_coco.py $WORK_DIR/mask_rcnn_r50_fpn_groie_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION ga_faster_r50_caffe_fpn_1x_coco configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py $WORK_DIR/ga_faster_r50_caffe_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster_rcnn_hrnetv2p_w18_1x_coco configs/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco.py $WORK_DIR/faster_rcnn_hrnetv2p_w18_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/htc/htc_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION htc_r50_fpn_1x_coco configs/htc/htc_r50_fpn_1x_coco.py $WORK_DIR/htc_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask_rcnn_r50_fpn_instaboost_4x_coco configs/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco.py $WORK_DIR/mask_rcnn_r50_fpn_instaboost_4x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/lad/lad_r50_paa_r101_fpn_coco_1x.py' &
GPUS=2  GPUS_PER_NODE=2  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION lad_r50_paa_r101_fpn_coco_1x configs/lad/lad_r50_paa_r101_fpn_coco_1x.py $WORK_DIR/lad_r50_paa_r101_fpn_coco_1x --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/ld/ld_r18_gflv1_r101_fpn_coco_1x.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION ld_r18_gflv1_r101_fpn_coco_1x configs/ld/ld_r18_gflv1_r101_fpn_coco_1x.py $WORK_DIR/ld_r18_gflv1_r101_fpn_coco_1x --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION libra_faster_rcnn_r50_fpn_1x_coco configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py $WORK_DIR/libra_faster_rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask2former_r50_lsj_8x2_50e_coco configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py $WORK_DIR/mask2former_r50_lsj_8x2_50e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py $WORK_DIR/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/maskformer/maskformer_r50_mstrain_16x1_75e_coco.py' &
GPUS=16  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION maskformer_r50_mstrain_16x1_75e_coco configs/maskformer/maskformer_r50_mstrain_16x1_75e_coco.py $WORK_DIR/maskformer_r50_mstrain_16x1_75e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION ms_rcnn_r50_caffe_fpn_1x_coco configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py $WORK_DIR/ms_rcnn_r50_caffe_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py $WORK_DIR/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_r50_nasfpn_crop640_50e_coco configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py $WORK_DIR/retinanet_r50_nasfpn_crop640_50e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/paa/paa_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION paa_r50_fpn_1x_coco configs/paa/paa_r50_fpn_1x_coco.py $WORK_DIR/paa_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/pafpn/faster_rcnn_r50_pafpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster_rcnn_r50_pafpn_1x_coco configs/pafpn/faster_rcnn_r50_pafpn_1x_coco.py $WORK_DIR/faster_rcnn_r50_pafpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION panoptic_fpn_r50_fpn_1x_coco configs/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco.py $WORK_DIR/panoptic_fpn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/pisa/pisa_mask_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION pisa_mask_rcnn_r50_fpn_1x_coco configs/pisa/pisa_mask_rcnn_r50_fpn_1x_coco.py $WORK_DIR/pisa_mask_rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION point_rend_r50_caffe_fpn_mstrain_1x_coco configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py $WORK_DIR/point_rend_r50_caffe_fpn_mstrain_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/pvt/retinanet_pvt-t_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_pvt-t_fpn_1x_coco configs/pvt/retinanet_pvt-t_fpn_1x_coco.py $WORK_DIR/retinanet_pvt-t_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/queryinst/queryinst_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION queryinst_r50_fpn_1x_coco configs/queryinst/queryinst_r50_fpn_1x_coco.py $WORK_DIR/queryinst_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/regnet/retinanet_regnetx-800MF_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_regnetx-800MF_fpn_1x_coco configs/regnet/retinanet_regnetx-800MF_fpn_1x_coco.py $WORK_DIR/retinanet_regnetx-800MF_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION reppoints_moment_r50_fpn_gn-neck+head_1x_coco configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py $WORK_DIR/reppoints_moment_r50_fpn_gn-neck+head_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/res2net/faster_rcnn_r2_101_fpn_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster_rcnn_r2_101_fpn_2x_coco configs/res2net/faster_rcnn_r2_101_fpn_2x_coco.py $WORK_DIR/faster_rcnn_r2_101_fpn_2x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/resnest/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco configs/resnest/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py $WORK_DIR/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/resnet_strikes_back/retinanet_r50_fpn_rsb-pretrain_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_r50_fpn_rsb-pretrain_1x_coco configs/resnet_strikes_back/retinanet_r50_fpn_rsb-pretrain_1x_coco.py $WORK_DIR/retinanet_r50_fpn_rsb-pretrain_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/retinanet/retinanet_r50_caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_r50_caffe_fpn_1x_coco configs/retinanet/retinanet_r50_caffe_fpn_1x_coco.py $WORK_DIR/retinanet_r50_caffe_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/rpn/rpn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION rpn_r50_fpn_1x_coco configs/rpn/rpn_r50_fpn_1x_coco.py $WORK_DIR/rpn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/sabl/sabl_retinanet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION sabl_retinanet_r50_fpn_1x_coco configs/sabl/sabl_retinanet_r50_fpn_1x_coco.py $WORK_DIR/sabl_retinanet_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/scnet/scnet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION scnet_r50_fpn_1x_coco configs/scnet/scnet_r50_fpn_1x_coco.py $WORK_DIR/scnet_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster_rcnn_r50_fpn_gn-all_scratch_6x_coco configs/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco.py $WORK_DIR/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/solo/solo_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION solo_r50_fpn_1x_coco configs/solo/solo_r50_fpn_1x_coco.py $WORK_DIR/solo_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/solov2/solov2_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION solov2_r50_fpn_1x_coco configs/solov2/solov2_r50_fpn_1x_coco.py $WORK_DIR/solov2_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION sparse_rcnn_r50_fpn_1x_coco configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py $WORK_DIR/sparse_rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/ssd/ssd300_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION ssd300_coco configs/ssd/ssd300_coco.py $WORK_DIR/ssd300_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION ssdlite_mobilenetv2_scratch_600e_coco configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py $WORK_DIR/ssdlite_mobilenetv2_scratch_600e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask_rcnn_swin-t-p4-w7_fpn_1x_coco configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py $WORK_DIR/mask_rcnn_swin-t-p4-w7_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/tood/tood_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION tood_r50_fpn_1x_coco configs/tood/tood_r50_fpn_1x_coco.py $WORK_DIR/tood_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/tridentnet/tridentnet_r50_caffe_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION tridentnet_r50_caffe_1x_coco configs/tridentnet/tridentnet_r50_caffe_1x_coco.py $WORK_DIR/tridentnet_r50_caffe_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/vfnet/vfnet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION vfnet_r50_fpn_1x_coco configs/vfnet/vfnet_r50_fpn_1x_coco.py $WORK_DIR/vfnet_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/yolact/yolact_r50_8x8_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION yolact_r50_8x8_coco configs/yolact/yolact_r50_8x8_coco.py $WORK_DIR/yolact_r50_8x8_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/yolo/yolov3_d53_320_273e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION yolov3_d53_320_273e_coco configs/yolo/yolov3_d53_320_273e_coco.py $WORK_DIR/yolov3_d53_320_273e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/yolof/yolof_r50_c5_8x8_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION yolof_r50_c5_8x8_1x_coco configs/yolof/yolof_r50_c5_8x8_1x_coco.py $WORK_DIR/yolof_r50_c5_8x8_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/yolox/yolox_tiny_8x8_300e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION yolox_tiny_8x8_300e_coco configs/yolox/yolox_tiny_8x8_300e_coco.py $WORK_DIR/yolox_tiny_8x8_300e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
