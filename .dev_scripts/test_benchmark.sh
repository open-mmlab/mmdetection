PARTITION=$1
CHECKPOINT_DIR=$2

echo 'configs/atss/atss_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION atss_r50_fpn_1x_coco configs/atss/atss_r50_fpn_1x_coco.py $CHECKPOINT_DIR/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth --work-dir tools/batch_test/atss_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29666  &
echo 'configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION autoassign_r50_fpn_8x2_1x_coco configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py $CHECKPOINT_DIR/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth --work-dir tools/batch_test/autoassign_r50_fpn_8x2_1x_coco --eval bbox --cfg-option dist_params.port=29667  &
echo 'configs/carafe/faster_rcnn_r50_fpn_carafe_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION faster_rcnn_r50_fpn_carafe_1x_coco configs/carafe/faster_rcnn_r50_fpn_carafe_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.386_20200504_175733-385a75b7.pth --work-dir tools/batch_test/faster_rcnn_r50_fpn_carafe_1x_coco --eval bbox --cfg-option dist_params.port=29668  &
echo 'configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION cascade_rcnn_r50_fpn_1x_coco configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth --work-dir tools/batch_test/cascade_rcnn_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29669  &
echo 'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION cascade_mask_rcnn_r50_fpn_1x_coco configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth --work-dir tools/batch_test/cascade_mask_rcnn_r50_fpn_1x_coco --eval bbox segm --cfg-option dist_params.port=29670  &
echo 'configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION crpn_faster_rcnn_r50_caffe_fpn_1x_coco configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py $CHECKPOINT_DIR/crpn_faster_rcnn_r50_caffe_fpn_1x_coco-c8283cca.pth --work-dir tools/batch_test/crpn_faster_rcnn_r50_caffe_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29671  &
echo 'configs/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION centripetalnet_hourglass104_mstest_16x6_210e_coco configs/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py $CHECKPOINT_DIR/centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804-3ccc61e5.pth --work-dir tools/batch_test/centripetalnet_hourglass104_mstest_16x6_210e_coco --eval bbox --cfg-option dist_params.port=29672  &
echo 'configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION cornernet_hourglass104_mstest_8x6_210e_coco configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py $CHECKPOINT_DIR/cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618-79b44c30.pth --work-dir tools/batch_test/cornernet_hourglass104_mstest_8x6_210e_coco --eval bbox --cfg-option dist_params.port=29673  &
echo 'configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth --work-dir tools/batch_test/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco --eval bbox --cfg-option dist_params.port=29674  &
echo 'configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION deformable_detr_r50_16x2_50e_coco configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py $CHECKPOINT_DIR/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth --work-dir tools/batch_test/deformable_detr_r50_16x2_50e_coco --eval bbox --cfg-option dist_params.port=29675  &
echo 'configs/detectors/detectors_htc_r50_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION detectors_htc_r50_1x_coco configs/detectors/detectors_htc_r50_1x_coco.py $CHECKPOINT_DIR/detectors_htc_r50_1x_coco-329b1453.pth --work-dir tools/batch_test/detectors_htc_r50_1x_coco --eval bbox segm --cfg-option dist_params.port=29676  &
echo 'configs/detr/detr_r50_8x2_150e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION detr_r50_8x2_150e_coco configs/detr/detr_r50_8x2_150e_coco.py $CHECKPOINT_DIR/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth --work-dir tools/batch_test/detr_r50_8x2_150e_coco --eval bbox --cfg-option dist_params.port=29677  &
echo 'configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION dh_faster_rcnn_r50_fpn_1x_coco configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth --work-dir tools/batch_test/dh_faster_rcnn_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29678  &
echo 'configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION dynamic_rcnn_r50_fpn_1x_coco configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/dynamic_rcnn_r50_fpn_1x-62a3f276.pth --work-dir tools/batch_test/dynamic_rcnn_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29679  &
echo 'configs/empirical_attention/faster_rcnn_r50_fpn_attention_1111_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION faster_rcnn_r50_fpn_attention_1111_1x_coco configs/empirical_attention/faster_rcnn_r50_fpn_attention_1111_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_fpn_attention_1111_1x_coco_20200130-403cccba.pth --work-dir tools/batch_test/faster_rcnn_r50_fpn_attention_1111_1x_coco --eval bbox --cfg-option dist_params.port=29680  &
echo 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION faster_rcnn_r50_fpn_1x_coco configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --work-dir tools/batch_test/faster_rcnn_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29681  &
echo 'configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py $CHECKPOINT_DIR/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth --work-dir tools/batch_test/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco --eval bbox --cfg-option dist_params.port=29682  &
echo 'configs/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION fovea_align_r50_fpn_gn-head_4x4_2x_coco configs/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py $CHECKPOINT_DIR/fovea_align_r50_fpn_gn-head_4x4_2x_coco_20200203-8987880d.pth --work-dir tools/batch_test/fovea_align_r50_fpn_gn-head_4x4_2x_coco --eval bbox --cfg-option dist_params.port=29683  &
echo 'configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION retinanet_free_anchor_r50_fpn_1x_coco configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py $CHECKPOINT_DIR/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth --work-dir tools/batch_test/retinanet_free_anchor_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29684  &
echo 'configs/fsaf/fsaf_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION fsaf_r50_fpn_1x_coco configs/fsaf/fsaf_r50_fpn_1x_coco.py $CHECKPOINT_DIR/fsaf_r50_fpn_1x_coco-94ccc51f.pth --work-dir tools/batch_test/fsaf_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29685  &
echo 'configs/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco configs/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco.py $CHECKPOINT_DIR/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200202-587b99aa.pth --work-dir tools/batch_test/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco --eval bbox segm --cfg-option dist_params.port=29686  &
echo 'configs/gfl/gfl_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION gfl_r50_fpn_1x_coco configs/gfl/gfl_r50_fpn_1x_coco.py $CHECKPOINT_DIR/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth --work-dir tools/batch_test/gfl_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29687  &
echo 'configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION mask_rcnn_r50_fpn_gn-all_2x_coco configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py $CHECKPOINT_DIR/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth --work-dir tools/batch_test/mask_rcnn_r50_fpn_gn-all_2x_coco --eval bbox segm --cfg-option dist_params.port=29688  &
echo 'configs/gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION faster_rcnn_r50_fpn_gn_ws-all_1x_coco configs/gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth --work-dir tools/batch_test/faster_rcnn_r50_fpn_gn_ws-all_1x_coco --eval bbox --cfg-option dist_params.port=29689  &
echo 'configs/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION grid_rcnn_r50_fpn_gn-head_2x_coco configs/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py $CHECKPOINT_DIR/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth --work-dir tools/batch_test/grid_rcnn_r50_fpn_gn-head_2x_coco --eval bbox --cfg-option dist_params.port=29690  &
echo 'configs/groie/faster_rcnn_r50_fpn_groie_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION faster_rcnn_r50_fpn_groie_1x_coco configs/groie/faster_rcnn_r50_fpn_groie_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_fpn_groie_1x_coco_20200604_211715-66ee9516.pth --work-dir tools/batch_test/faster_rcnn_r50_fpn_groie_1x_coco --eval bbox --cfg-option dist_params.port=29691  &
echo 'configs/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION ga_retinanet_r50_caffe_fpn_1x_coco configs/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco.py $CHECKPOINT_DIR/ga_retinanet_r50_caffe_fpn_1x_coco_20201020-39581c6f.pth --work-dir tools/batch_test/ga_retinanet_r50_caffe_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29692  &
echo 'configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION ga_faster_r50_caffe_fpn_1x_coco configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py $CHECKPOINT_DIR/ga_faster_r50_caffe_fpn_1x_coco_20200702_000718-a11ccfe6.pth --work-dir tools/batch_test/ga_faster_r50_caffe_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29693  &
echo 'configs/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION faster_rcnn_hrnetv2p_w18_1x_coco configs/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_hrnetv2p_w18_1x_coco_20200130-56651a6d.pth --work-dir tools/batch_test/faster_rcnn_hrnetv2p_w18_1x_coco --eval bbox --cfg-option dist_params.port=29694  &
echo 'configs/htc/htc_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION htc_r50_fpn_1x_coco configs/htc/htc_r50_fpn_1x_coco.py $CHECKPOINT_DIR/htc_r50_fpn_1x_coco_20200317-7332cf16.pth --work-dir tools/batch_test/htc_r50_fpn_1x_coco --eval bbox segm --cfg-option dist_params.port=29695  &
echo 'configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION libra_faster_rcnn_r50_fpn_1x_coco configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth --work-dir tools/batch_test/libra_faster_rcnn_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29696  &
echo 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION mask_rcnn_r50_fpn_1x_coco configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth --work-dir tools/batch_test/mask_rcnn_r50_fpn_1x_coco --eval bbox segm --cfg-option dist_params.port=29697  &
echo 'configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION ms_rcnn_r50_caffe_fpn_1x_coco configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py $CHECKPOINT_DIR/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth --work-dir tools/batch_test/ms_rcnn_r50_caffe_fpn_1x_coco --eval bbox segm --cfg-option dist_params.port=29698  &
echo 'configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py $CHECKPOINT_DIR/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200520-1bdba3ce.pth --work-dir tools/batch_test/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco --eval bbox --cfg-option dist_params.port=29699  &
echo 'configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION retinanet_r50_nasfpn_crop640_50e_coco configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py $CHECKPOINT_DIR/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth --work-dir tools/batch_test/retinanet_r50_nasfpn_crop640_50e_coco --eval bbox --cfg-option dist_params.port=29700  &
echo 'configs/paa/paa_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION paa_r50_fpn_1x_coco configs/paa/paa_r50_fpn_1x_coco.py $CHECKPOINT_DIR/paa_r50_fpn_1x_coco_20200821-936edec3.pth --work-dir tools/batch_test/paa_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29701  &
echo 'configs/pafpn/faster_rcnn_r50_pafpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION faster_rcnn_r50_pafpn_1x_coco configs/pafpn/faster_rcnn_r50_pafpn_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_r50_pafpn_1x_coco_bbox_mAP-0.375_20200503_105836-b7b4b9bd.pth --work-dir tools/batch_test/faster_rcnn_r50_pafpn_1x_coco --eval bbox --cfg-option dist_params.port=29702  &
echo 'configs/pisa/pisa_faster_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION pisa_faster_rcnn_r50_fpn_1x_coco configs/pisa/pisa_faster_rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/pisa_faster_rcnn_r50_fpn_1x_coco-dea93523.pth --work-dir tools/batch_test/pisa_faster_rcnn_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29703  &
echo 'configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION point_rend_r50_caffe_fpn_mstrain_1x_coco configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py $CHECKPOINT_DIR/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth --work-dir tools/batch_test/point_rend_r50_caffe_fpn_mstrain_1x_coco --eval bbox segm --cfg-option dist_params.port=29704  &
echo 'configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION mask_rcnn_regnetx-3.2GF_fpn_1x_coco configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py $CHECKPOINT_DIR/mask_rcnn_regnetx-3.2GF_fpn_1x_coco_20200520_163141-2a9d1814.pth --work-dir tools/batch_test/mask_rcnn_regnetx-3.2GF_fpn_1x_coco --eval bbox segm --cfg-option dist_params.port=29705  &
echo 'configs/reppoints/reppoints_moment_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION reppoints_moment_r50_fpn_1x_coco configs/reppoints/reppoints_moment_r50_fpn_1x_coco.py $CHECKPOINT_DIR/reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth --work-dir tools/batch_test/reppoints_moment_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29706  &
echo 'configs/res2net/faster_rcnn_r2_101_fpn_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION faster_rcnn_r2_101_fpn_2x_coco configs/res2net/faster_rcnn_r2_101_fpn_2x_coco.py $CHECKPOINT_DIR/faster_rcnn_r2_101_fpn_2x_coco-175f1da6.pth --work-dir tools/batch_test/faster_rcnn_r2_101_fpn_2x_coco --eval bbox --cfg-option dist_params.port=29707  &
echo 'configs/resnest/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco configs/resnest/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py $CHECKPOINT_DIR/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco_20200926_125502-20289c16.pth --work-dir tools/batch_test/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco --eval bbox --cfg-option dist_params.port=29708  &
echo 'configs/retinanet/retinanet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION retinanet_r50_fpn_1x_coco configs/retinanet/retinanet_r50_fpn_1x_coco.py $CHECKPOINT_DIR/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --work-dir tools/batch_test/retinanet_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29709  &
echo 'configs/rpn/rpn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION rpn_r50_fpn_1x_coco configs/rpn/rpn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth --work-dir tools/batch_test/rpn_r50_fpn_1x_coco --eval proposal_fast --cfg-option dist_params.port=29710  &
echo 'configs/sabl/sabl_retinanet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION sabl_retinanet_r50_fpn_1x_coco configs/sabl/sabl_retinanet_r50_fpn_1x_coco.py $CHECKPOINT_DIR/sabl_retinanet_r50_fpn_1x_coco-6c54fd4f.pth --work-dir tools/batch_test/sabl_retinanet_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29711  &
echo 'configs/sabl/sabl_faster_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION sabl_faster_rcnn_r50_fpn_1x_coco configs/sabl/sabl_faster_rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/sabl_faster_rcnn_r50_fpn_1x_coco-e867595b.pth --work-dir tools/batch_test/sabl_faster_rcnn_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29712  &
echo 'configs/scnet/scnet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION scnet_r50_fpn_1x_coco configs/scnet/scnet_r50_fpn_1x_coco.py $CHECKPOINT_DIR/scnet_r50_fpn_1x_coco-c3f09857.pth --work-dir tools/batch_test/scnet_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29713  &
echo 'configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION sparse_rcnn_r50_fpn_1x_coco configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py $CHECKPOINT_DIR/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth --work-dir tools/batch_test/sparse_rcnn_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29714  &
echo 'configs/ssd/ssd300_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION ssd300_coco configs/ssd/ssd300_coco.py $CHECKPOINT_DIR/ssd300_coco_20210803_015428-d231a06e.pth --work-dir tools/batch_test/ssd300_coco --eval bbox --cfg-option dist_params.port=29715  &
echo 'configs/tridentnet/tridentnet_r50_caffe_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION tridentnet_r50_caffe_1x_coco configs/tridentnet/tridentnet_r50_caffe_1x_coco.py $CHECKPOINT_DIR/tridentnet_r50_caffe_1x_coco_20201230_141838-2ec0b530.pth --work-dir tools/batch_test/tridentnet_r50_caffe_1x_coco --eval bbox --cfg-option dist_params.port=29716  &
echo 'configs/vfnet/vfnet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION vfnet_r50_fpn_1x_coco configs/vfnet/vfnet_r50_fpn_1x_coco.py $CHECKPOINT_DIR/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth --work-dir tools/batch_test/vfnet_r50_fpn_1x_coco --eval bbox --cfg-option dist_params.port=29717  &
echo 'configs/yolact/yolact_r50_1x8_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION yolact_r50_1x8_coco configs/yolact/yolact_r50_1x8_coco.py $CHECKPOINT_DIR/yolact_r50_1x8_coco_20200908-f38d58df.pth --work-dir tools/batch_test/yolact_r50_1x8_coco --eval bbox segm --cfg-option dist_params.port=29718  &
echo 'configs/yolo/yolov3_d53_320_273e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION yolov3_d53_320_273e_coco configs/yolo/yolov3_d53_320_273e_coco.py $CHECKPOINT_DIR/yolov3_d53_320_273e_coco-421362b6.pth --work-dir tools/batch_test/yolov3_d53_320_273e_coco --eval bbox --cfg-option dist_params.port=29719  &
echo 'configs/yolof/yolof_r50_c5_8x8_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION yolof_r50_c5_8x8_1x_coco configs/yolof/yolof_r50_c5_8x8_1x_coco.py $CHECKPOINT_DIR/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth --work-dir tools/batch_test/yolof_r50_c5_8x8_1x_coco --eval bbox --cfg-option dist_params.port=29720  &
echo 'configs/centernet/centernet_resnet18_dcnv2_140e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION centernet_resnet18_dcnv2_140e_coco configs/centernet/centernet_resnet18_dcnv2_140e_coco.py $CHECKPOINT_DIR/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth --work-dir tools/batch_test/centernet_resnet18_dcnv2_140e_coco --eval bbox --cfg-option dist_params.port=29721  &
echo 'configs/yolox/yolox_tiny_8x8_300e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION yolox_tiny_8x8_300e_coco configs/yolox/yolox_tiny_8x8_300e_coco.py $CHECKPOINT_DIR/yolox_tiny_8x8_300e_coco_20210806_234250-4ff3b67e.pth --work-dir tools/batch_test/yolox_tiny_8x8_300e_coco --eval bbox --cfg-option dist_params.port=29722  &
echo 'configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION ssdlite_mobilenetv2_scratch_600e_coco configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py $CHECKPOINT_DIR/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth --work-dir tools/batch_test/ssdlite_mobilenetv2_scratch_600e_coco --eval bbox --cfg-option dist_params.port=29723  &
