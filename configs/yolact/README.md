# **Y**ou **O**nly **L**ook **A**t **C**oefficien**T**s
```
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝
```

A simple, fully convolutional model for real-time instance segmentation. This is the code for our paper:
 - [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)
 <!-- - [YOLACT++: Better Real-time Instance Segmentation](https://arxiv.org/abs/1912.06218) -->

#### For a real-time demo, check out our ICCV video:
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/0pMfmo8qfpQ/0.jpg)](https://www.youtube.com/watch?v=0pMfmo8qfpQ)

# Evaluation
Here are our YOLACT models along with their FPS on a Titan Xp and mAP on COCO's `val`:

| Image Size | GPU x BS | Backbone      | *FPS  | mAP  | Weights |
|:----------:|:--------:|:-------------:|:-----:|:----:|---------|
| 550        | 1x8      | Resnet50-FPN  | 42.5 | 29.0 | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/yolact/yolact_r50_1x8_coco_20200908-f38d58df.pth) |
| 550        | 8x8      | Resnet50-FPN  | 42.5 | 28.4 | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/yolact/yolact_r50_8x8_coco_20200908-ca34f5db.pth) |
| 550        | 1x8      | Resnet101-FPN | 33.5 | 30.4 | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/yolact/yolact_r101_1x8_coco_20200908-4cbe9101.pth) |

*Note: The FPS is evaluated by the [original implementation](https://github.com/dbolya/yolact). When calculating FPS, only the model inference time is taken into account. Data loading and post-processing operations such as converting masks to RLE code, generating COCO JSON results, image rendering are not included.

# Training
All the aforementioned models are trained with a single GPU. It typically takes ~12GB VRAM when using resnet-101 as the backbone. If you want to try multiple GPUs training, you may have to modify the configuration files accordingly, such as adjusting the training schedule and freezing batch norm.
```Shell
# Trains using the resnet-101 backbone with a batch size of 8 on a single GPU.
./tools/dist_train.sh configs/yolact/yolact_r101.py 1
```

# Testing
Please refer to [mmdetection/docs/getting_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/getting_started.md#inference-with-pretrained-models).

# Citation
If you use YOLACT or this code base in your work, please cite
```
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
```

<!-- For YOLACT++, please cite
```
@misc{yolact-plus-arxiv2019,
  title         = {YOLACT++: Better Real-time Instance Segmentation},
  author        = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  year          = {2019},
  eprint        = {1912.06218},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
``` -->
