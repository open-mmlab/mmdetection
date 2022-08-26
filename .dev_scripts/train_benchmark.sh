PARTITION=$1
WORK_DIR=$2
CPUS_PER_TASK=${3:-4}

echo 'configs/albu_example/mask-rcnn_r50_fpn_albu_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask-rcnn_r50_fpn_albu_1x_coco configs/albu_example/mask-rcnn_r50_fpn_albu_1x_coco.py $WORK_DIR/mask-rcnn_r50_fpn_albu_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/atss/atss_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION atss_r50_fpn_1x_coco configs/atss/atss_r50_fpn_1x_coco.py $WORK_DIR/atss_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/autoassign/autoassign_r50-caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION autoassign_r50-caffe_fpn_1x_coco configs/autoassign/autoassign_r50-caffe_fpn_1x_coco.py $WORK_DIR/autoassign_r50-caffe_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/carafe/faster-rcnn_r50_fpn-carafe_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster-rcnn_r50_fpn-carafe_1x_coco configs/carafe/faster-rcnn_r50_fpn-carafe_1x_coco.py $WORK_DIR/faster-rcnn_r50_fpn-carafe_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION cascade-rcnn_r50_fpn_1x_coco configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py $WORK_DIR/cascade-rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION cascade-mask-rcnn_r50_fpn_1x_coco configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py $WORK_DIR/cascade-mask-rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/cascade_rpn/cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco configs/cascade_rpn/cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco.py $WORK_DIR/cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/centernet/centernet_r18-dcnv2_8xb16-crop512-140e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION centernet_r18-dcnv2_8xb16-crop512-140e_coco configs/centernet/centernet_r18-dcnv2_8xb16-crop512-140e_coco.py $WORK_DIR/centernet_r18-dcnv2_8xb16-crop512-140e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION centernet-update_r50-caffe_fpn_ms-1x_coco configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py $WORK_DIR/centernet-update_r50-caffe_fpn_ms-1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py' &
GPUS=16  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco configs/centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py $WORK_DIR/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION cornernet_hourglass104_8xb6-210e-mstest_coco configs/cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py $WORK_DIR/cornernet_hourglass104_8xb6-210e-mstest_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco configs/convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py $WORK_DIR/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/dcn/faster-rcnn_r50-dconv-c3-c5_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster-rcnn_r50-dconv-c3-c5_fpn_1x_coco configs/dcn/faster-rcnn_r50-dconv-c3-c5_fpn_1x_coco.py $WORK_DIR/faster-rcnn_r50-dconv-c3-c5_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/dcnv2/faster-rcnn_r50_fpn_mdpool_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster-rcnn_r50_fpn_mdpool_1x_coco configs/dcnv2/faster-rcnn_r50_fpn_mdpool_1x_coco.py $WORK_DIR/faster-rcnn_r50_fpn_mdpool_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/ddod/ddod_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION ddod_r50_fpn_1x_coco configs/ddod/ddod_r50_fpn_1x_coco.py $WORK_DIR/ddod_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/detectors/detectors_htc-r50_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION detectors_htc-r50_1x_coco configs/detectors/detectors_htc-r50_1x_coco.py $WORK_DIR/detectors_htc-r50_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py' &
GPUS=16  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION deformable-detr_r50_16xb2-50e_coco configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py $WORK_DIR/deformable-detr_r50_16xb2-50e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/detr/detr_r50_8xb2-150e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION detr_r50_8xb2-150e_coco configs/detr/detr_r50_8xb2-150e_coco.py $WORK_DIR/detr_r50_8xb2-150e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION dh-faster-rcnn_r50_fpn_1x_coco configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py $WORK_DIR/dh-faster-rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION dynamic-rcnn_r50_fpn_1x_coco configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py $WORK_DIR/dynamic-rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION atss_r50_fpn_dyhead_1x_coco configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py $WORK_DIR/atss_r50_fpn_dyhead_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/efficientnet/retinanet_effb3_fpn_8xb4-crop896-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_effb3_fpn_8xb4-crop896-1x_coco configs/efficientnet/retinanet_effb3_fpn_8xb4-crop896-1x_coco.py $WORK_DIR/retinanet_effb3_fpn_8xb4-crop896-1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/empirical_attention/faster-rcnn_r50-attn1111_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster-rcnn_r50-attn1111_fpn_1x_coco configs/empirical_attention/faster-rcnn_r50-attn1111_fpn_1x_coco.py $WORK_DIR/faster-rcnn_r50-attn1111_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster-rcnn_r50_fpn_1x_coco configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py $WORK_DIR/faster-rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/faster_rcnn/faster-rcnn_r50-caffe-dc5_ms-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster-rcnn_r50-caffe-dc5_ms-1x_coco configs/faster_rcnn/faster-rcnn_r50-caffe-dc5_ms-1x_coco.py $WORK_DIR/faster-rcnn_r50-caffe-dc5_ms-1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/fcos/fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco configs/fcos/fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py $WORK_DIR/fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/foveabox/fovea_r50_fpn_gn-head-align_4xb4-2x_coco.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION fovea_r50_fpn_gn-head-align_4xb4-2x_coco configs/foveabox/fovea_r50_fpn_gn-head-align_4xb4-2x_coco.py $WORK_DIR/fovea_r50_fpn_gn-head-align_4xb4-2x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/fpg/mask-rcnn_r50_fpg_crop640-50e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask-rcnn_r50_fpg_crop640-50e_coco configs/fpg/mask-rcnn_r50_fpg_crop640-50e_coco.py $WORK_DIR/mask-rcnn_r50_fpg_crop640-50e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/free_anchor/freeanchor_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION freeanchor_r50_fpn_1x_coco configs/free_anchor/freeanchor_r50_fpn_1x_coco.py $WORK_DIR/freeanchor_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/fsaf/fsaf_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION fsaf_r50_fpn_1x_coco configs/fsaf/fsaf_r50_fpn_1x_coco.py $WORK_DIR/fsaf_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/gcnet/mask-rcnn_r50-syncbn-gcb-r16-c3-c5_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask-rcnn_r50-syncbn-gcb-r16-c3-c5_fpn_1x_coco configs/gcnet/mask-rcnn_r50-syncbn-gcb-r16-c3-c5_fpn_1x_coco.py $WORK_DIR/mask-rcnn_r50-syncbn-gcb-r16-c3-c5_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/gfl/gfl_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION gfl_r50_fpn_1x_coco configs/gfl/gfl_r50_fpn_1x_coco.py $WORK_DIR/gfl_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/ghm/retinanet_r50_fpn_ghm-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_r50_fpn_ghm-1x_coco configs/ghm/retinanet_r50_fpn_ghm-1x_coco.py $WORK_DIR/retinanet_r50_fpn_ghm-1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/gn/mask-rcnn_r50_fpn_gn-all_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask-rcnn_r50_fpn_gn-all_2x_coco configs/gn/mask-rcnn_r50_fpn_gn-all_2x_coco.py $WORK_DIR/mask-rcnn_r50_fpn_gn-all_2x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/gn+ws/faster-rcnn_r50_fpn_gn-ws-all_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster-rcnn_r50_fpn_gn-ws-all_1x_coco configs/gn+ws/faster-rcnn_r50_fpn_gn-ws-all_1x_coco.py $WORK_DIR/faster-rcnn_r50_fpn_gn-ws-all_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION grid-rcnn_r50_fpn_gn-head_2x_coco configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_coco.py $WORK_DIR/grid-rcnn_r50_fpn_gn-head_2x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/groie/faste-rcnn_r50_fpn_groie_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faste-rcnn_r50_fpn_groie_1x_coco configs/groie/faste-rcnn_r50_fpn_groie_1x_coco.py $WORK_DIR/faste-rcnn_r50_fpn_groie_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/guided_anchoring/ga-retinanet_r50-caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION ga-retinanet_r50-caffe_fpn_1x_coco configs/guided_anchoring/ga-retinanet_r50-caffe_fpn_1x_coco.py $WORK_DIR/ga-retinanet_r50-caffe_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/hrnet/faster-rcnn_hrnetv2p-w18-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster-rcnn_hrnetv2p-w18-1x_coco configs/hrnet/faster-rcnn_hrnetv2p-w18-1x_coco.py $WORK_DIR/faster-rcnn_hrnetv2p-w18-1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/htc/htc_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION htc_r50_fpn_1x_coco configs/htc/htc_r50_fpn_1x_coco.py $WORK_DIR/htc_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/instaboost/mask-rcnn_r50_fpn_instaboost-4x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask-rcnn_r50_fpn_instaboost-4x_coco configs/instaboost/mask-rcnn_r50_fpn_instaboost-4x_coco.py $WORK_DIR/mask-rcnn_r50_fpn_instaboost-4x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/lad/lad_r50-paa-r101_fpn_2xb8_coco_1x.py' &
GPUS=2  GPUS_PER_NODE=2  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION lad_r50-paa-r101_fpn_2xb8_coco_1x configs/lad/lad_r50-paa-r101_fpn_2xb8_coco_1x.py $WORK_DIR/lad_r50-paa-r101_fpn_2xb8_coco_1x --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/ld/ld_r18-gflv1-r101_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION ld_r18-gflv1-r101_fpn_1x_coco configs/ld/ld_r18-gflv1-r101_fpn_1x_coco.py $WORK_DIR/ld_r18-gflv1-r101_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION libra-faster-rcnn_r50_fpn_1x_coco configs/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py $WORK_DIR/libra-faster-rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask2former_r50_8xb2-lsj-50e_coco-panoptic configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py $WORK_DIR/mask2former_r50_8xb2-lsj-50e_coco-panoptic --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask-rcnn_r50_fpn_1x_coco configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py $WORK_DIR/mask-rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/maskformer/maskformer_r50_ms-16xb1-75e_coco.py' &
GPUS=16  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION maskformer_r50_ms-16xb1-75e_coco configs/maskformer/maskformer_r50_ms-16xb1-75e_coco.py $WORK_DIR/maskformer_r50_ms-16xb1-75e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/ms_rcnn/ms-rcnn_r50-caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION ms-rcnn_r50-caffe_fpn_1x_coco configs/ms_rcnn/ms-rcnn_r50-caffe_fpn_1x_coco.py $WORK_DIR/ms-rcnn_r50-caffe_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/nas_fcos/nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco configs/nas_fcos/nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco.py $WORK_DIR/nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/nas_fpn/retinanet_r50_nasfpn_crop640-50e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_r50_nasfpn_crop640-50e_coco configs/nas_fpn/retinanet_r50_nasfpn_crop640-50e_coco.py $WORK_DIR/retinanet_r50_nasfpn_crop640-50e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/paa/paa_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION paa_r50_fpn_1x_coco configs/paa/paa_r50_fpn_1x_coco.py $WORK_DIR/paa_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/pafpn/faster-rcnn_r50_pafpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster-rcnn_r50_pafpn_1x_coco configs/pafpn/faster-rcnn_r50_pafpn_1x_coco.py $WORK_DIR/faster-rcnn_r50_pafpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/panoptic_fpn/panoptic-fpn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION panoptic-fpn_r50_fpn_1x_coco configs/panoptic_fpn/panoptic-fpn_r50_fpn_1x_coco.py $WORK_DIR/panoptic-fpn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/pisa/mask-rcnn_r50_fpn_pisa_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask-rcnn_r50_fpn_pisa_1x_coco configs/pisa/mask-rcnn_r50_fpn_pisa_1x_coco.py $WORK_DIR/mask-rcnn_r50_fpn_pisa_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION point-rend_r50-caffe_fpn_ms-1x_coco configs/point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py $WORK_DIR/point-rend_r50-caffe_fpn_ms-1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/pvt/retinanet_pvt-t_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_pvt-t_fpn_1x_coco configs/pvt/retinanet_pvt-t_fpn_1x_coco.py $WORK_DIR/retinanet_pvt-t_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/queryinst/queryinst_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION queryinst_r50_fpn_1x_coco configs/queryinst/queryinst_r50_fpn_1x_coco.py $WORK_DIR/queryinst_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/regnet/retinanet_regnetx-800MF_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_regnetx-800MF_fpn_1x_coco configs/regnet/retinanet_regnetx-800MF_fpn_1x_coco.py $WORK_DIR/retinanet_regnetx-800MF_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/reppoints/reppoints-moment_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION reppoints-moment_r50_fpn_1x_coco configs/reppoints/reppoints-moment_r50_fpn_1x_coco.py $WORK_DIR/reppoints-moment_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/res2net/faster-rcnn_res2net-101_fpn_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster-rcnn_res2net-101_fpn_2x_coco configs/res2net/faster-rcnn_res2net-101_fpn_2x_coco.py $WORK_DIR/faster-rcnn_res2net-101_fpn_2x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/resnest/faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco configs/resnest/faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco.py $WORK_DIR/faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/resnet_strikes_back/retinanet_r50-rsb-pre_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_r50-rsb-pre_fpn_1x_coco configs/resnet_strikes_back/retinanet_r50-rsb-pre_fpn_1x_coco.py $WORK_DIR/retinanet_r50-rsb-pre_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/retinanet/retinanet_r50-caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION retinanet_r50-caffe_fpn_1x_coco configs/retinanet/retinanet_r50-caffe_fpn_1x_coco.py $WORK_DIR/retinanet_r50-caffe_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/rpn/rpn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION rpn_r50_fpn_1x_coco configs/rpn/rpn_r50_fpn_1x_coco.py $WORK_DIR/rpn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/sabl/sabl-retinanet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION sabl-retinanet_r50_fpn_1x_coco configs/sabl/sabl-retinanet_r50_fpn_1x_coco.py $WORK_DIR/sabl-retinanet_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/scnet/scnet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION scnet_r50_fpn_1x_coco configs/scnet/scnet_r50_fpn_1x_coco.py $WORK_DIR/scnet_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/scratch/faster-rcnn_r50-scratch_fpn_gn-all_6x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION faster-rcnn_r50-scratch_fpn_gn-all_6x_coco configs/scratch/faster-rcnn_r50-scratch_fpn_gn-all_6x_coco.py $WORK_DIR/faster-rcnn_r50-scratch_fpn_gn-all_6x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/solo/solo_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION solo_r50_fpn_1x_coco configs/solo/solo_r50_fpn_1x_coco.py $WORK_DIR/solo_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/solov2/solov2_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION solov2_r50_fpn_1x_coco configs/solov2/solov2_r50_fpn_1x_coco.py $WORK_DIR/solov2_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION sparse-rcnn_r50_fpn_1x_coco configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py $WORK_DIR/sparse-rcnn_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/ssd/ssd300_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION ssd300_coco configs/ssd/ssd300_coco.py $WORK_DIR/ssd300_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION ssdlite_mobilenetv2-scratch_8xb24-600e_coco configs/ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py $WORK_DIR/ssdlite_mobilenetv2-scratch_8xb24-600e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask-rcnn_swin-t-p4-w7_fpn_1x_coco configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py $WORK_DIR/mask-rcnn_swin-t-p4-w7_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/tood/tood_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION tood_r50_fpn_1x_coco configs/tood/tood_r50_fpn_1x_coco.py $WORK_DIR/tood_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo ''configs/tridentnet/tridentnet_r50-caffe_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION tridentnet_r50-caffe_1x_coco 'configs/tridentnet/tridentnet_r50-caffe_1x_coco.py $WORK_DIR/tridentnet_r50-caffe_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/vfnet/vfnet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION vfnet_r50_fpn_1x_coco configs/vfnet/vfnet_r50_fpn_1x_coco.py $WORK_DIR/vfnet_r50_fpn_1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/yolact/yolact_r50_8xb8-55e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION yolact_r50_8xb8-55e_coco configs/yolact/yolact_r50_8xb8-55e_coco.py $WORK_DIR/yolact_r50_8xb8-55e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/yolo/yolov3_d53_8xb8-320-273e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION yolov3_d53_8xb8-320-273e_coco configs/yolo/yolov3_d53_8xb8-320-273e_coco.py $WORK_DIR/yolov3_d53_8xb8-320-273e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/yolof/yolof_r50-c5_8xb8-1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION yolof_r50-c5_8xb8-1x_coco configs/yolof/yolof_r50-c5_8xb8-1x_coco.py $WORK_DIR/yolof_r50-c5_8xb8-1x_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/yolox/yolox_tiny_8xb8-300e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION yolox_tiny_8xb8-300e_coco configs/yolox/yolox_tiny_8xb8-300e_coco.py $WORK_DIR/yolox_tiny_8xb8-300e_coco --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
