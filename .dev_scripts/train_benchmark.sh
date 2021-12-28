echo 'configs/atss/atss_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab atss_r50_fpn_1x_coco configs/atss/atss_r50_fpn_1x_coco.py ./tools/work_dir/atss_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab autoassign_r50_fpn_8x2_1x_coco configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py ./tools/work_dir/autoassign_r50_fpn_8x2_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab cascade_mask_rcnn_r50_fpn_1x_coco configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py ./tools/work_dir/cascade_mask_rcnn_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab crpn_faster_rcnn_r50_caffe_fpn_1x_coco configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py ./tools/work_dir/crpn_faster_rcnn_r50_caffe_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/centernet/centernet_resnet18_dcnv2_140e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab centernet_resnet18_dcnv2_140e_coco configs/centernet/centernet_resnet18_dcnv2_140e_coco.py ./tools/work_dir/centernet_resnet18_dcnv2_140e_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py' &
GPUS=16  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab centripetalnet_hourglass104_mstest_16x6_210e_coco configs/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py ./tools/work_dir/centripetalnet_hourglass104_mstest_16x6_210e_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab cornernet_hourglass104_mstest_8x6_210e_coco configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py ./tools/work_dir/cornernet_hourglass104_mstest_8x6_210e_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/detectors/detectors_htc_r50_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab detectors_htc_r50_1x_coco configs/detectors/detectors_htc_r50_1x_coco.py ./tools/work_dir/detectors_htc_r50_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py' &
GPUS=16  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab deformable_detr_r50_16x2_50e_coco configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py ./tools/work_dir/deformable_detr_r50_16x2_50e_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/detr/detr_r50_8x2_150e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab detr_r50_8x2_150e_coco configs/detr/detr_r50_8x2_150e_coco.py ./tools/work_dir/detr_r50_8x2_150e_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab dh_faster_rcnn_r50_fpn_1x_coco configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py ./tools/work_dir/dh_faster_rcnn_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab dynamic_rcnn_r50_fpn_1x_coco configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py ./tools/work_dir/dynamic_rcnn_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab faster_rcnn_r50_fpn_1x_coco configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py ./tools/work_dir/faster_rcnn_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab faster_rcnn_r50_caffe_dc5_mstrain_1x_coco configs/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco.py ./tools/work_dir/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab faster_rcnn_r50_caffe_fpn_mstrain_1x_coco configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py ./tools/work_dir/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab faster_rcnn_r50_caffe_fpn_1x_coco configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco.py ./tools/work_dir/faster_rcnn_r50_caffe_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab faster_rcnn_r50_fpn_ohem_1x_coco configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py ./tools/work_dir/faster_rcnn_r50_fpn_ohem_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab fovea_align_r50_fpn_gn-head_4x4_2x_coco configs/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py ./tools/work_dir/fovea_align_r50_fpn_gn-head_4x4_2x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/fp16/mask_rcnn_r50_fpn_fp16_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab mask_rcnn_r50_fpn_fp16_1x_coco configs/fp16/mask_rcnn_r50_fpn_fp16_1x_coco.py ./tools/work_dir/mask_rcnn_r50_fpn_fp16_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/fp16/retinanet_r50_fpn_fp16_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab retinanet_r50_fpn_fp16_1x_coco configs/fp16/retinanet_r50_fpn_fp16_1x_coco.py ./tools/work_dir/retinanet_r50_fpn_fp16_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab retinanet_free_anchor_r50_fpn_1x_coco configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py ./tools/work_dir/retinanet_free_anchor_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/fsaf/fsaf_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab fsaf_r50_fpn_1x_coco configs/fsaf/fsaf_r50_fpn_1x_coco.py ./tools/work_dir/fsaf_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/gfl/gfl_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab gfl_r50_fpn_1x_coco configs/gfl/gfl_r50_fpn_1x_coco.py ./tools/work_dir/gfl_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/ghm/retinanet_ghm_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab retinanet_ghm_r50_fpn_1x_coco configs/ghm/retinanet_ghm_r50_fpn_1x_coco.py ./tools/work_dir/retinanet_ghm_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab grid_rcnn_r50_fpn_gn-head_2x_coco configs/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py ./tools/work_dir/grid_rcnn_r50_fpn_gn-head_2x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab ga_faster_r50_caffe_fpn_1x_coco configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py ./tools/work_dir/ga_faster_r50_caffe_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/htc/htc_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab htc_r50_fpn_1x_coco configs/htc/htc_r50_fpn_1x_coco.py ./tools/work_dir/htc_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/ld/ld_r18_gflv1_r101_fpn_coco_1x.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab ld_r18_gflv1_r101_fpn_coco_1x configs/ld/ld_r18_gflv1_r101_fpn_coco_1x.py ./tools/work_dir/ld_r18_gflv1_r101_fpn_coco_1x --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab libra_faster_rcnn_r50_fpn_1x_coco configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py ./tools/work_dir/libra_faster_rcnn_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py ./tools/work_dir/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab ms_rcnn_r50_caffe_fpn_1x_coco configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py ./tools/work_dir/ms_rcnn_r50_caffe_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py ./tools/work_dir/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/paa/paa_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab paa_r50_fpn_1x_coco configs/paa/paa_r50_fpn_1x_coco.py ./tools/work_dir/paa_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/pisa/pisa_mask_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab pisa_mask_rcnn_r50_fpn_1x_coco configs/pisa/pisa_mask_rcnn_r50_fpn_1x_coco.py ./tools/work_dir/pisa_mask_rcnn_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab point_rend_r50_caffe_fpn_mstrain_1x_coco configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py ./tools/work_dir/point_rend_r50_caffe_fpn_mstrain_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab reppoints_moment_r50_fpn_gn-neck+head_1x_coco configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py ./tools/work_dir/reppoints_moment_r50_fpn_gn-neck+head_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/retinanet/retinanet_r50_caffe_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab retinanet_r50_caffe_fpn_1x_coco configs/retinanet/retinanet_r50_caffe_fpn_1x_coco.py ./tools/work_dir/retinanet_r50_caffe_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/rpn/rpn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab rpn_r50_fpn_1x_coco configs/rpn/rpn_r50_fpn_1x_coco.py ./tools/work_dir/rpn_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/sabl/sabl_retinanet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab sabl_retinanet_r50_fpn_1x_coco configs/sabl/sabl_retinanet_r50_fpn_1x_coco.py ./tools/work_dir/sabl_retinanet_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/ssd/ssd300_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab ssd300_coco configs/ssd/ssd300_coco.py ./tools/work_dir/ssd300_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/tridentnet/tridentnet_r50_caffe_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab tridentnet_r50_caffe_1x_coco configs/tridentnet/tridentnet_r50_caffe_1x_coco.py ./tools/work_dir/tridentnet_r50_caffe_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/vfnet/vfnet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab vfnet_r50_fpn_1x_coco configs/vfnet/vfnet_r50_fpn_1x_coco.py ./tools/work_dir/vfnet_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/yolact/yolact_r50_8x8_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab yolact_r50_8x8_coco configs/yolact/yolact_r50_8x8_coco.py ./tools/work_dir/yolact_r50_8x8_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/yolo/yolov3_d53_320_273e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab yolov3_d53_320_273e_coco configs/yolo/yolov3_d53_320_273e_coco.py ./tools/work_dir/yolov3_d53_320_273e_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab sparse_rcnn_r50_fpn_1x_coco configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py ./tools/work_dir/sparse_rcnn_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/scnet/scnet_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab scnet_r50_fpn_1x_coco configs/scnet/scnet_r50_fpn_1x_coco.py ./tools/work_dir/scnet_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/yolof/yolof_r50_c5_8x8_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab yolof_r50_c5_8x8_1x_coco configs/yolof/yolof_r50_c5_8x8_1x_coco.py ./tools/work_dir/yolof_r50_c5_8x8_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/carafe/mask_rcnn_r50_fpn_carafe_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab mask_rcnn_r50_fpn_carafe_1x_coco configs/carafe/mask_rcnn_r50_fpn_carafe_1x_coco.py ./tools/work_dir/mask_rcnn_r50_fpn_carafe_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab faster_rcnn_r50_fpn_mdpool_1x_coco configs/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco.py ./tools/work_dir/faster_rcnn_r50_fpn_mdpool_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco configs/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py ./tools/work_dir/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/dcn/faster_rcnn_r50_fpn_dpool_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab faster_rcnn_r50_fpn_dpool_1x_coco configs/dcn/faster_rcnn_r50_fpn_dpool_1x_coco.py ./tools/work_dir/faster_rcnn_r50_fpn_dpool_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/dcn/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco configs/dcn/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco.py ./tools/work_dir/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco configs/empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco.py ./tools/work_dir/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py ./tools/work_dir/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab mask_rcnn_r50_fpn_gn-all_2x_coco configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py ./tools/work_dir/mask_rcnn_r50_fpn_gn-all_2x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/gn+ws/mask_rcnn_r50_fpn_gn_ws-all_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab mask_rcnn_r50_fpn_gn_ws-all_2x_coco configs/gn+ws/mask_rcnn_r50_fpn_gn_ws-all_2x_coco.py ./tools/work_dir/mask_rcnn_r50_fpn_gn_ws-all_2x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab mask_rcnn_hrnetv2p_w18_1x_coco configs/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco.py ./tools/work_dir/mask_rcnn_hrnetv2p_w18_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/pafpn/faster_rcnn_r50_pafpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab faster_rcnn_r50_pafpn_1x_coco configs/pafpn/faster_rcnn_r50_pafpn_1x_coco.py ./tools/work_dir/faster_rcnn_r50_pafpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab retinanet_r50_nasfpn_crop640_50e_coco configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py ./tools/work_dir/retinanet_r50_nasfpn_crop640_50e_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab mask_rcnn_regnetx-3.2GF_fpn_1x_coco configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py ./tools/work_dir/mask_rcnn_regnetx-3.2GF_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/resnest/mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco configs/resnest/mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py ./tools/work_dir/mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/res2net/faster_rcnn_r2_101_fpn_2x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab faster_rcnn_r2_101_fpn_2x_coco configs/res2net/faster_rcnn_r2_101_fpn_2x_coco.py ./tools/work_dir/faster_rcnn_r2_101_fpn_2x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/groie/faster_rcnn_r50_fpn_groie_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab faster_rcnn_r50_fpn_groie_1x_coco configs/groie/faster_rcnn_r50_fpn_groie_1x_coco.py ./tools/work_dir/faster_rcnn_r50_fpn_groie_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab mask_rcnn_r50_fpn_1x_cityscapes configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py ./tools/work_dir/mask_rcnn_r50_fpn_1x_cityscapes --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab panoptic_fpn_r50_fpn_1x_coco configs/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco.py ./tools/work_dir/panoptic_fpn_r50_fpn_1x_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/yolox/yolox_tiny_8x8_300e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab yolox_tiny_8x8_300e_coco configs/yolox/yolox_tiny_8x8_300e_coco.py ./tools/work_dir/yolox_tiny_8x8_300e_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
echo 'configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab ssdlite_mobilenetv2_scratch_600e_coco configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py ./tools/work_dir/ssdlite_mobilenetv2_scratch_600e_coco --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
