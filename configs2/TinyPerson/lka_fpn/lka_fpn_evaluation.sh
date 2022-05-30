export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_lka_fpn_noaem_noffm_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g_noaem_noffm/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

export GPU=4 && LR=0.04 CONFIG="faster_rcnn_r50_lka_fpn_noaem_noffm_1x_TinyPerson640_newData"
python demo/image_demo.py data/tiny_set/test/labeled_dense_images/baidu_P000_24.jpg \
 configs2/TinyPerson/lka_fpn/${CONFIG}.py \
../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g_noaem_noffm/epoch_12.pth
