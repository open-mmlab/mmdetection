# Done image [850/ 2000], fps: 2.4 img / s, times per image: 413.8 ms / img 大概一致都是这个速率
export GPU=2 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_1x_TinyPerson640_newData"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
       configs2/TinyPerson/lka_fpn/${CONFIG}.py \
       ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g_withbnrelu_withsigmoid_withlastatt/epoch_11.pth \
       --launcher pytorch

# baseline低
# Done image [150/ 2000], fps: 3.2 img / s, times per image: 309.1 ms / img 原版差不多是这个速率
export GPU=4 && LR=0.016 && CONFIG="faster_rcnn_r50_fpn_1x_TinyPerson640_newData"
CUDA_VISIABLE_DEVICE=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
       configs2/TinyPerson/base/${CONFIG}.py \
       ../TOV_mmdetection_cache/work_dir/TinyPerson/base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/epoch_12.pth \
       --launcher pytorch



