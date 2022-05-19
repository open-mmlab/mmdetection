# aptiny50 0.5822 tiny1 0.4414 tiny2 0.6213 tiny3 0.6724 small 0.7320
export GPU=2 && CONFIG="faster_rcnn_r50_carafe_fpn_1x_TinyPerson640_newData" && CUDA_VISIBLE_DEVICES=2,3 PORT=10011 \
   tools/dist_test.sh configs2/TinyPerson/carafe/${CONFIG}.py \
   ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/${CONFIG}/old640x512_lr0.008_1x_4g/latest.pth \
   ${GPU} --work-dir exp/${CONFIG} --eval bbox
