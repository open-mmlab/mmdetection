# Scale Match
# exp4.0 coco-sm-tinyperson: Faster-FPN, coco batch=8x2
export GPU=2 && export LR=0.01 && export BATCH=8 && export CONFIG="faster_rcnn_r50_fpn_1x_coco_sm_tinyperson"
export COCO_WORK_DIR="../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/${CONFIG}/lr${LR}_1x_${BATCH}b${GPU}g/"
CUDA_VISIBLE_DEVICES=0,1 PORT=10001 tools/dist_train.sh configs2/TinyPerson/scale_match/${CONFIG}.py $GPU \
  --work-dir ${COCO_WORK_DIR} --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${BATCH}
# python exp/tools/extract_weight.py ${COCO_WORK_DIR}/latest.pth
export TCONFIG="faster_rcnn_r50_fpn_1x_TinyPerson640"
export GPU=2 && LR=0.01 && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/TinyPerson/base/${TCONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/${TCONFIG}/coco8x2bsm_old640x512_lr${LR}_1x_${GPU}g/ \
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