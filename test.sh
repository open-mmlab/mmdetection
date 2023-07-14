#!/bin/bash
#此参数用于指定运行作业的名称
#DSUB -n test
#此处需要把“用户名”修改为用户的用户名，例如用户名为 gpuuser001 则此行写为“#DSUB -A root.bingxing2.gpuuse206”
#DSUB -A root.bingxing2.gpuuser206
#默认参数，一般不需要修改 
#DSUB -q root.default
#DSUB -l wuhanG5500
#跨节点任务不同类型程序 job_type 会有差异，请参考下文对应跨节点任务模板编写
#DSUB --job_type cosched
#此参数用于指定资源。如申请 6 核 CPU，1 卡 GPU，48GB 内存。
#DSUB -R 'cpu=48;gpu=8;mem=360000'
#此参数用于指定运行作业的机器数量。单节点作业则为 1 。
#DSUB -N 1
# 此参数用于指定日志的输出，%J 表示 JOB_ID。
#DSUB -e %J.out
#DSUB -o %J.out
#加载环境，此处加载 anaconda 环境以及通过 anaconda 创建的名为 pytorch 的环境(pytorch 环境需要自己部署)
module load anaconda/2021.11 
source activate openmmlab
#python 运行程序
python tools/test.py projects/OneFormer/configs/oneformer_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco.py 150_16_swin_l_oneformer_coco_100ep.pth

python tools/test.py projects/OneFormer/configs/oneformer_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco_panoptic.py 150_16_swin_l_oneformer_coco_100ep.pth
# tools/dist_test.sh projects/OneFormer/configs/oneformer_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco.py 150_16_swin_l_oneformer_coco_100ep.pth

# tools/dist_test.sh projects/OneFormer/configs/oneformer_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco_panoptic.py 150_16_swin_l_oneformer_coco_100ep.pth 8