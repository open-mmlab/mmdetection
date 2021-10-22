
# exp1.1: Faster-FPN
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_r50_fpn_1x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# exp1.2: Faster-FPN, 2GPU
export GPU=2 && LR=0.01 && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_r50_fpn_1x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# exp2.1: adap RetinaNet
export GPU=1 && LR=0.005 && CONFIG="retinanet_r50_fpns4_1x_TinyPerson640" && CUDA_VISIBLE_DEVICES=0 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}
# # exp2.2: 4gpu clip grad
export GPU=4 && LR=0.02 && CONFIG="retinanet_r50_fpns4_1x_TinyPerson640"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/${CONFIG}/old640x512_lr${LR}_1x_clipg_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}
# # exp2.3: 2gpu clip grad
export GPU=2 && LR=0.01 && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/TinyPerson/base/retinanet_r50_fpns4_1x_TinyPerson640_clipg.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/retinanet_r50_fpns4_1x_TinyPerson640/old640x512_lr${LR}_1x_clipg_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}

# exp3.1 RepPoint w/o GN neck, backbone norm no grad => easy to nan
export GPU=2 && LR=0.01 && CONFIG="reppoints_moment_r50_fpn_1x_TinyPerson640" && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 \
  tools/dist_train.sh configs2/TinyPerson/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} model.backbone.norm_cfg.requires_grad=False

# exp3.2 RepPoint w/o GN neck => Nan
export GPU=2 && LR=0.01 && CONFIG="reppoints_moment_r50_fpn_1x_TinyPerson640" && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 \
  tools/dist_train.sh configs2/TinyPerson/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# exp3.3 Adap RepPoint w/o GN neck, backbone norm no grad => Nan
export GPU=2 && LR=0.01 && CONFIG="reppoints_moment_r50_fpns4_1x_TinyPerson640" && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 \
  tools/dist_train.sh configs2/TinyPerson/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# exp3.4 RepPoint
export GPU=2 && LR=0.01 && CONFIG="reppoints_moment_r50_fpn_gn-neck+head_1x_TinyPerson640" && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 \
  tools/dist_train.sh configs2/TinyPerson/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# exp3.6 RepPoint, backbone norm no grad
export GPU=2 && LR=0.01 && CONFIG="reppoints_moment_r50_fpn_gn-neck+head_1x_TinyPerson640" && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 \
  tools/dist_train.sh configs2/TinyPerson/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} model.backbone.norm_cfg.requires_grad=False

# exp3.5 Adap RepPoint
export GPU=2 && LR=0.01 && CONFIG="reppoints_moment_r50_fpns4_gn-neck+head_1x_TinyPerson640" && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 \
  tools/dist_train.sh configs2/TinyPerson/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# Scale Match
# exp4.0 coco-sm-tinyperson: Faster-FPN, coco batch=8x2
export GPU=2 && export LR=0.01 && export BATCH=8 && export CONFIG="faster_rcnn_r50_fpn_1x_coco_sm_tinyperson"
export COCO_WORK_DIR="../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/${CONFIG}/lr${LR}_1x_${BATCH}b${GPU}g/"
CUDA_VISIBLE_DEVICES=0,1 PORT=10001 tools/dist_train.sh configs2/TinyPerson/scale_match/${CONFIG}.py $GPU \
  --work-dir ${COCO_WORK_DIR} --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${BATCH}
# python exp/tools/extract_weight.py ${COCO_WORK_DIR}/latest.pth
export TCONFIG="faster_rcnn_r50_fpn_1x_TinyPerson640"
export GPU=2 && LR=0.01 && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/TinyPerson/base/${TCONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/${TCONFIG}/cocosm_old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} load_from=${COCO_WORK_DIR}/latest.pth

# exp4.1 coco-sm-tinyperson: Faster-FPN, coco batch=4x2
export GPU=2 && export LR=0.01 && export BATCH=4 && export CONFIG="faster_rcnn_r50_fpn_1x_coco_sm_tinyperson"
export COCO_WORK_DIR="../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/${CONFIG}/lr${LR}_1x_${BATCH}b${GPU}g/"
CUDA_VISIBLE_DEVICES=0,1 PORT=10001 tools/dist_train.sh configs2/TinyPerson/scale_match/${CONFIG}.py $GPU \
  --work-dir ${COCO_WORK_DIR} --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${BATCH}

export TCONFIG="faster_rcnn_r50_fpn_1x_TinyPerson640"
export GPU=2 && LR=0.01 && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/TinyPerson/base/${TCONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/${TCONFIG}/cocosm_old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} load_from=${COCO_WORK_DIR}/latest.pth

# exp4.2 coco-msm-tinyperson: Faster-FPN
export GPU=2 && export LR=0.01 && export BATCH=4 && export CONFIG="faster_rcnn_r50_fpn_1x_coco_msm_tinyperson"
export COCO_WORK_DIR="../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/${CONFIG}/lr${LR}_1x_${BATCH}b${GPU}g/"
CUDA_VISIBLE_DEVICES=0,1 PORT=10001 tools/dist_train.sh configs2/TinyPerson/scale_match/${CONFIG}.py $GPU \
  --work-dir ${COCO_WORK_DIR} --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${BATCH}

export TCONFIG="faster_rcnn_r50_fpn_1x_TinyPerson640"
export GPU=2 && LR=0.01 && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/TinyPerson/base/${TCONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/${TCONFIG}/cocomsm_old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} load_from=${COCO_WORK_DIR}/latest.pth

# exp5.1 coco-sm-tinyperson: Adap RetinaNet
export GPU=2 && export LR=0.01 && export BATCH=4 && export CONFIG="retinanet_r50_fpns4_1x_coco_sm_tinyperson"
export COCO_WORK_DIR="../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/${CONFIG}/lr${LR}_1x_${BATCH}b${GPU}g/"
CUDA_VISIBLE_DEVICES=0,1 PORT=10001 tools/dist_train.sh configs2/TinyPerson/scale_match/${CONFIG}.py $GPU \
  --work-dir ${COCO_WORK_DIR} --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${BATCH}
export GPU=1 && LR=005 && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/TinyPerson/base/retinanet_r50_fpns4_1x_TinyPerson640_clipg.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/retinanet_r50_fpns4_1x_TinyPerson640_clipg/cocosm_old640x512_lr0${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=0.${LR} load_from=${COCO_WORK_DIR}/latest.pth

# exp5.2 coco-msm-tinyperson: Adap RetinaNet
export GPU=2 && export LR=0.01 && export BATCH=4 && export CONFIG="retinanet_r50_fpns4_1x_coco_msm_tinyperson"
export COCO_WORK_DIR="../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/${CONFIG}/lr${LR}_1x_${BATCH}b${GPU}g/"
CUDA_VISIBLE_DEVICES=0,1 PORT=10001 tools/dist_train.sh configs2/TinyPerson/scale_match/${CONFIG}.py $GPU \
  --work-dir ${COCO_WORK_DIR} --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${BATCH}
export GPU=2 && LR=0.01 && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/TinyPerson/base/retinanet_r50_fpns4_1x_TinyPerson640_clipg.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/retinanet_r50_fpns4_1x_TinyPerson640_clipg/cocomsm_old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} load_from=${COCO_WORK_DIR}/latest.pth



# exp6.1 Adap FCOS
export GPU=2 && LR=0.01 && CONFIG="fcos_standard_r50_caffe_fpns4_gn-head_1x_TinyPerson640" && CUDA_VISIBLE_DEVICES=0,1 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}
