# Benchmark and Model Zoo

## Environment

### Hardware

- 8 NVIDIA Tesla V100 GPUs
- Intel Xeon 4114 CPU @ 2.20GHz

### Software environment

- Python 3.6 / 3.7
- PyTorch 1.0
- CUDA 9.0.176
- CUDNN 7.0.4
- NCCL 2.1.15

Note: The train time was measured with PyTorch 0.4.1. We will update it later, which should be about 0.02s ~ 0.05s faster.

## Mirror sites

We use AWS as the main site to host our model zoo, and maintain a mirror on aliyun.
You can replace `https://s3.ap-northeast-2.amazonaws.com/open-mmlab` with `https://open-mmlab.oss-cn-beijing.aliyuncs.com` in model urls.

## Common settings

- All baselines were trained using 8 GPU with a batch size of 16 (2 images per GPU).
- All models were trained on `coco_2017_train`, and tested on the `coco_2017_val`.
- We use distributed training and BN layer stats are fixed.
- We adopt the same training schedules as Detectron. 1x indicates 12 epochs and 2x indicates 24 epochs, which corresponds to slightly less iterations than Detectron and the difference can be ignored.
- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo.
- We report the training GPU memory as the maximum value of `torch.cuda.max_memory_cached()`
for all 8 GPUs. Note that this value is usually less than what `nvidia-smi` shows, but
closer to the actual requirements.
- We report the inference time as the overall time including data loading,
network forwarding and post processing.
- The training memory and time of 2x schedule is simply copied from 1x.
It should be very close to the actual memory and time.


## Baselines

We released RPN, Faster R-CNN and Mask R-CNN models in the first version. More models with different backbones will be added to the model zoo.

### RPN

| Backbone | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | AR1000 | Download |
|:--------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50-FPN | caffe   | 1x      | 4.5      | 0.379               | 14.4           | 58.2   | -        |
| R-50-FPN | pytorch | 1x      | 4.8      | 0.407               | 14.5           | 57.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_fpn_1x_20181010-4a9c0712.pth) |
| R-50-FPN | pytorch | 2x      | 4.8      | 0.407               | 14.5           | 57.6   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_fpn_2x_20181010-88a4a471.pth) |
| R-101-FPN | caffe   | 1x      | 7.4      | 0.513               | 11.1           | 59.4   | -        |
| R-101-FPN | pytorch | 1x      | 8.0      | 0.552               | 11.1           | 58.6   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r101_fpn_1x_20181129-f50da4bd.pth) |
| R-101-FPN | pytorch | 2x      | 8.0      | 0.552               | 11.1           | 59.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r101_fpn_2x_20181129-e42c6c9a.pth) |
| X-101-32x4d-FPN | pytorch |1x | 9.9      | 0.691               | 8.3            | 59.4   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_x101_32x4d_fpn_1x_20181218-7e379d26.pth)
| X-101-32x4d-FPN | pytorch |2x | 9.9      | 0.691               | 8.3            | 59.9   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_x101_32x4d_fpn_2x_20181218-0510af40.pth)
| X-101-64x4d-FPN | pytorch |1x | 14.6     | 1.032               | 6.2            | 59.8   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_x101_64x4d_fpn_1x_20181218-c1a24f1f.pth)
| X-101-64x4d-FPN | pytorch |2x | 14.6     | 1.032               | 6.2            | 60.0   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_x101_64x4d_fpn_2x_20181218-c22bdd70.pth)

### Faster R-CNN

| Backbone | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:--------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50-FPN | caffe   | 1x      | 4.9      | 0.525               | 10.0           | 36.7   | -        |
| R-50-FPN | pytorch | 1x      | 5.1      | 0.554               | 9.9            | 36.4   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth) |
| R-50-FPN | pytorch | 2x      | 5.1      | 0.554               | 9.9            | 37.7   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth) |
| R-101-FPN | caffe   | 1x      | 7.4      | 0.663               | 8.4           | 38.8   | -        |
| R-101-FPN | pytorch | 1x      | 8.0      | 0.698               | 8.3           | 38.6   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r101_fpn_1x_20181129-d1468807.pth) |
| R-101-FPN | pytorch | 2x      | 8.0      | 0.698               | 8.3           | 39.4   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r101_fpn_2x_20181129-73e7ade7.pth) |
| X-101-32x4d-FPN | pytorch | 1x| 9.9      | 0.842               | 7.0           | 40.2    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_32x4d_fpn_1x_20181218-ad81c133.pth)
| X-101-32x4d-FPN | pytorch | 2x| 9.9      | 0.842               | 7.0           | 40.5    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_32x4d_fpn_2x_20181218-0ed58946.pth)
| X-101-64x4d-FPN | pytorch | 1x| 14.1     | 1.181               | 5.2           | 41.3    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_64x4d_fpn_1x_20181218-c9c69c8f.pth)
| X-101-64x4d-FPN | pytorch | 2x| 14.1     | 1.181               | 5.2           | 40.7    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_64x4d_fpn_2x_20181218-fe94f9b8.pth)

### Mask R-CNN

| Backbone | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
|:--------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
| R-50-FPN | caffe   | 1x      | 5.9      | 0.658               | 7.7            | 37.5   | 34.4    | -        |
| R-50-FPN | pytorch | 1x      | 5.8      | 0.690               | 7.7            | 37.3   | 34.2    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth) |
| R-50-FPN | pytorch | 2x      | 5.8      | 0.690               | 7.7            | 38.6   | 35.1    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth) |
| R-101-FPN | caffe   | 1x      | 8.8      | 0.791               | 7.0            | 39.9   | 36.1    | -        |
| R-101-FPN | pytorch | 1x      | 9.1      | 0.825               | 6.7            | 39.4   | 35.9    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r101_fpn_1x_20181129-34ad1961.pth) |
| R-101-FPN | pytorch | 2x      | 9.1      | 0.825               | 6.7            | 40.4   | 36.6    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r101_fpn_2x_20181129-a254bdfc.pth) |
| X-101-32x4d-FPN | pytorch | 1x| 10.9     | 0.972               | 5.8            | 41.2   | 37.2    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_x101_32x4d_fpn_1x_20181218-44e635cc.pth)
| X-101-64x4d-FPN | pytorch | 2x| 10.9     | 0.972               | 5.8            | 41.4   | 37.1    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_x101_32x4d_fpn_2x_20181218-f023dffa.pth)
| X-101-32x4d-FPN | pytorch | 1x| 14.1     | 1.302               | 4.7            | 42.2   | 38.1    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_x101_64x4d_fpn_1x_20181218-cb159987.pth)
| X-101-64x4d-FPN | pytorch | 2x| 14.1     | 1.302               | 4.7            | 42.0   | 37.8    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_x101_64x4d_fpn_2x_20181218-ea936e44.pth)

### Fast R-CNN (with pre-computed proposals)

| Backbone | Style   | Type   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
|:--------:|:-------:|:------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
| R-50-FPN | caffe   | Faster | 1x      | 3.5      | 0.348               | 14.6           | 36.6   | -       | -        |
| R-50-FPN | pytorch | Faster | 1x      | 4.0      | 0.375               | 14.5           | 35.8   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_rcnn_r50_fpn_1x_20181010-08160859.pth) |
| R-50-FPN | pytorch | Faster | 2x      | 4.0      | 0.375               | 14.5           | 37.1   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_rcnn_r50_fpn_2x_20181010-d263ada5.pth) |
| R-101-FPN| caffe   | Faster | 1x      | 7.1      | 0.484               | 11.9           | 38.4   | -       | -        |
| R-101-FPN| pytorch | Faster | 1x      | 7.6      | 0.540               | 11.8           | 38.1   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_rcnn_r101_fpn_1x_20181129-ffaa2eb0.pth) |
| R-101-FPN| pytorch | Faster | 2x      | 7.6      | 0.540               | 11.8           | 38.8   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_rcnn_r101_fpn_2x_20181129-9dba92ce.pth) |
| R-50-FPN | caffe   | Mask   | 1x      | 5.4      | 0.473               | 10.7           | 37.3   | 34.5    | -        |
| R-50-FPN | pytorch | Mask   | 1x      | 5.3      | 0.504               | 10.6           | 36.8   | 34.1    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_mask_rcnn_r50_fpn_1x_20181010-e030a38f.pth) |
| R-50-FPN | pytorch | Mask   | 2x      | 5.3      | 0.504               | 10.6           | 37.9   | 34.8    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_mask_rcnn_r50_fpn_2x_20181010-5048cb03.pth) |
| R-101-FPN| caffe   | Mask   | 1x      | 8.6      | 0.607               | 9.5            | 39.4   | 36.1    | -        |
| R-101-FPN| pytorch | Mask   | 1x      | 9.0      | 0.656               | 9.3            | 38.9   | 35.8    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_mask_rcnn_r101_fpn_1x_20181129-2273fa9b.pth) |
| R-101-FPN| pytorch | Mask   | 2x      | 9.0      | 0.656               | 9.3            | 39.9   | 36.4    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_mask_rcnn_r101_fpn_2x_20181129-bf63ec5e.pth) |

### RetinaNet

| Backbone | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:--------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50-FPN | caffe   | 1x      | 6.7      | 0.468               | 9.4            | 35.8   | -        |
| R-50-FPN | pytorch | 1x      | 6.9      | 0.496               | 9.1            | 35.6   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-3d3c2142.pth) |
| R-50-FPN | pytorch | 2x      | 6.9      | 0.496               | 9.1            | 36.5   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_2x_20181125-e0dbec97.pth) |
| R-101-FPN | caffe   | 1x      | 9.2      | 0.614               | 8.2            | 37.8   | -        |
| R-101-FPN | pytorch | 1x      | 9.6      | 0.643               | 8.1            | 37.7   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r101_fpn_1x_20181129-f738a02f.pth) |
| R-101-FPN | pytorch | 2x      | 9.6      | 0.643               | 8.1            | 38.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r101_fpn_2x_20181129-f654534b.pth) |
| X-101-32x4d-FPN | pytorch | 1x| 10.8     | 0.792               | 6.7            | 38.7   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_32x4d_fpn_1x_20181218-c140fb82.pth)
| X-101-32x4d-FPN | pytorch | 2x| 10.8     | 0.792               | 6.7            | 39.3   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_32x4d_fpn_2x_20181218-605dcd0a.pth)
| X-101-64x4d-FPN | pytorch | 1x| 14.6     | 1.128               | 5.3            | 40.0   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_64x4d_fpn_1x_20181218-2f6f778b.pth)
| X-101-64x4d-FPN | pytorch | 2x| 14.6     | 1.128               | 5.3            | 39.6   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_64x4d_fpn_2x_20181218-2f598dc5.pth)

### Cascade R-CNN

| Backbone | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:--------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50-FPN | caffe   | 1x      | 5.0      | 0.592               | 8.1            | 40.3   | -        |
| R-50-FPN | pytorch | 1x      | 5.5      | 0.622               | 8.0            | 40.3   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r50_fpn_1x_20181123-b1987c4a.pth) |
| R-50-FPN | pytorch | 20e     | 5.5      | 0.622               | 8.0            | 41.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth) |
| R-101-FPN | caffe   | 1x      | 8.5      | 0.731               | 7.0            | 42.2   | -        |
| R-101-FPN | pytorch | 1x      | 8.7      | 0.766               | 6.9            | 42.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth) |
| R-101-FPN | pytorch | 20e     | 8.7      | 0.766               | 6.9            | 42.6   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r101_fpn_20e_20181129-b46dcede.pth) |
| X-101-32x4d-FPN | pytorch | 1x| 10.6     | 0.902               | 5.7            | 43.5   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_32x4d_fpn_1x_20181218-941c0925.pth)
| X-101-32x4d-FPN | pytorch |20e| 10.6     | 0.902               | 5.7            | 44.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_32x4d_fpn_2x_20181218-28f73c4c.pth)
| X-101-64x4d-FPN | pytorch | 1x| 14.1     | 1.251               | 4.6            | 44.6   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth)
| X-101-64x4d-FPN | pytorch |20e| 14.1     | 1.251               | 4.6            | 44.8   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_64x4d_fpn_2x_20181218-5add321e.pth)

### Cascade Mask R-CNN

| Backbone | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
|:--------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
| R-50-FPN | caffe   | 1x      | 7.5      | 0.880               | 5.8            | 41.0   | 35.6    | -        |
| R-50-FPN | pytorch | 1x      | 7.6      | 0.910               | 5.7            | 41.3   | 35.7    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r50_fpn_1x_20181123-88b170c9.pth) |
| R-50-FPN | pytorch | 20e     | 7.6      | 0.910               | 5.7            | 42.4   | 36.6    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r50_fpn_20e_20181123-6e0c9713.pth) |
| R-101-FPN | caffe   | 1x      | 10.5     | 1.024               | 5.3            | 43.1   | 37.3    | -        |
| R-101-FPN | pytorch | 1x      | 10.9     | 1.055               | 5.2            | 42.7   | 37.1    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r101_fpn_1x_20181129-64f00602.pth) |
| R-101-FPN | pytorch | 20e     | 10.9     | 1.055               | 5.2            | 43.4   | 37.6    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r101_fpn_20e_20181129-cb85151d.pth) |
| X-101-32x4d-FPN | pytorch | 1x| 12.67    | 1.181               | 4.2            | 44.4   | 38.3    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_x101_32x4d_fpn_1x_20181218-1d944c89.pth)
| X-101-32x4d-FPN | pytorch |20e| 12.67    | 1.181               | 4.2            | 44.9   | 38.7    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_x101_32x4d_fpn_20e_20181218-761a3473.pth)
| X-101-64x4d-FPN | pytorch | 1x| 10.87    | 1.125               | 3.6            | 45.5   | 39.2    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_x101_64x4d_fpn_1x_20181218-85953a91.pth)
| X-101-64x4d-FPN | pytorch |20e| 10.87    | 1.125               | 3.6            | 45.8   | 39.5    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_x101_64x4d_fpn_20e_20181218-630773a7.pth)

**Notes:**

- The `20e` schedule in Cascade (Mask) R-CNN indicates decreasing the lr at 16 and 19 epochs, with a total of 20 epochs.
- Cascade Mask R-CNN with X-101-64x4d-FPN was trained using 16 GPU with a batch size of 16 (1 images per GPU).

### SSD

| Backbone | Size | Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:--------:|:----:|:------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| VGG16    | 300  | caffe  | 120e    | 3.5      | 0.286               | 22.9 / 29.2    | 25.7   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth)  |
| VGG16    | 512  | caffe  | 120e    | 6.3      | 0.458               | 17.3 / 21.2    | 29.3   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ssd512_coco_vgg16_caffe_120e_20181221-d48b0be8.pth) |

### SSD (PASCAL VOC)

| Backbone | Size | Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:--------:|:----:|:------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| VGG16    | 300  | caffe  | 240e    | 1.2      | 0.189               | 40.1 / 58.0    | 77.8   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ssd300_voc_vgg16_caffe_240e_20181221-2f05dd40.pth)  |
| VGG16    | 512  | caffe  | 240e    | 2.9      | 0.261               | 28.1 / 36.2    | 80.4   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ssd512_voc_vgg16_caffe_240e_20181221-7652ee18.pth) |

**Notes:**

- `cudnn.benchmark` is set as `True` for SSD training and testing.
- Inference time is reported for batch size = 1 and batch size = 8.
- The speed difference between VOC and COCO is caused by model parameters and nms.

### Group Normalization (GN)

| Backbone      | model      | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
|:-------------:|:----------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
| R-50-FPN (d)  | Mask R-CNN | 2x      | 7.2      | 0.806               | 5.4            | 39.9   | 36.1    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_gn_2x_20180113-86832cf2.pth) |
| R-50-FPN (d)  | Mask R-CNN | 3x      | 7.2      | 0.806               | 5.4            | 40.2   | 36.5    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_gn_3x_20180113-8e82f48d.pth) |
| R-101-FPN (d) | Mask R-CNN | 2x      | 9.9      | 0.970               | 4.8            | 41.6   | 37.1    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r101_fpn_gn_2x_20180113-9598649c.pth) |
| R-101-FPN (d) | Mask R-CNN | 3x      | 9.9      | 0.970               | 4.8            | 41.7   | 37.3    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r101_fpn_gn_3x_20180113-a14ffb96.pth) |
| R-50-FPN (c)  | Mask R-CNN | 2x      | 7.2      | 0.806               | 5.4            | 39.7   | 35.9    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_gn_contrib_2x_20180113-ec93305c.pth) |
| R-50-FPN (c)  | Mask R-CNN | 3x      | 7.2      | 0.806               | 5.4            | 40.1   | 36.2    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_gn_contrib_3x_20180113-9d230cab.pth) |

**Notes:**
- (d) means pretrained model converted from Detectron, and (c) means the contributed model pretrained by [@thangvubk](https://github.com/thangvubk).
- The `3x` schedule is epoch [28, 34, 36].
- The memory is measured with `torch.cuda.max_memory_allocated()` instead of `torch.cuda.max_memory_cached()`. We will update the memory usage of other models in the future.

### Deformable Convolution v2

| Backbone  | Model        | Style   | Conv          | Pool   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
|:---------:|:------------:|:-------:|:-------------:|:------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
| R-50-FPN  | Faster       | pytorch | dconv(c3-c5)  | -      | 1x      | 3.9      | 0.594               | 10.2           | 40.0   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-e41688c9.pth) |
| R-50-FPN  | Faster       | pytorch | mdconv(c3-c5) | -      | 1x      | 3.7      | 0.598               | 10.0           | 40.3   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_mdconv_c3-c5_r50_fpn_1x_20190125-1b768045.pth) |
| R-50-FPN  | Faster       | pytorch | -             | dpool  | 1x      | 4.6      | 0.714               | 8.7            | 37.9   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_dpool_r50_fpn_1x_20190125-f4fc1d70.pth) |
| R-50-FPN  | Faster       | pytorch | -             | mdpool | 1x      | 5.2      | 0.769               | 8.2            | 38.1   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_mdpool_r50_fpn_1x_20190125-473d0f3d.pth) |
| R-101-FPN | Faster       | pytorch | dconv(c3-c5)  | -      | 1x      | 5.8      | 0.811               | 8.0            | 42.1   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-a7e31b65.pth) |
| X-101-32x4d-FPN | Faster       | pytorch | dconv(c3-c5)  | -      | 1x      | 7.1      | 1.126               | 6.6            | 43.5   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_20190201-6d46376f.pth) |
| R-50-FPN  | Mask         | pytorch | dconv(c3-c5)  | -      | 1x      | 4.5      | 0.712               | 7.7            | 41.1   | 37.2    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/mask_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-4f94ff79.pth) |
| R-50-FPN  | Mask         | pytorch | mdconv(c3-c5) | -      | 1x      | 4.5      | 0.712               | 7.7            | 41.4   | 37.4    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/mask_rcnn_mdconv_c3-c5_r50_fpn_1x_20190125-c5601dc3.pth) |
| R-101-FPN | Mask         | pytorch | dconv(c3-c5)  | -      | 1x      | 6.4      | 0.939               | 6.5            | 43.2   | 38.7    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/mask_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-decb6db5.pth) |
| R-50-FPN  | Cascade      | pytorch | dconv(c3-c5)  | -      | 1x      | 4.4      | 0.660               | 7.6            | 44.1   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166.pth) |
| R-101-FPN | Cascade      | pytorch | dconv(c3-c5)  | -      | 1x      | 6.3      | 0.881               | 6.8            | 45.1   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-aaa877cc.pth) |
| R-50-FPN  | Cascade Mask | pytorch | dconv(c3-c5)  | -      | 1x      | 6.6      | 0.942               | 5.7            | 44.5   | 38.3    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_mask_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-09d8a443.pth) |
| R-101-FPN | Cascade Mask | pytorch | dconv(c3-c5)  | -      | 1x      | 8.5      | 1.156               | 5.1            | 45.8   | 39.5    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_mask_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-0d62c190.pth) |

**Notes:**

- `dconv` and `mdconv` denote (modulated) deformable convolution, `c3-c5` means adding dconv in resnet stage 3 to 5. `dpool` and `mdpool` denote (modulated) deformable roi pooling.
- The memory is measured with `torch.cuda.max_memory_allocated()`. The batch size is 16 (2 images per GPU).
- The dcn ops are modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch, which should be more memory efficient and slightly faster.

## Comparison with Detectron

We compare mmdetection with [Detectron](https://github.com/facebookresearch/Detectron)
and [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch),
a third-party port of Detectron to Pytorch. The backbone used is R-50-FPN.

In general, mmdetection has 3 advantages over Detectron.

- **Higher performance** (especially in terms of mask AP)
- **Faster training speed**
- **Memory efficient**

### Performance

Detectron and Detectron.pytorch use caffe-style ResNet as the backbone.
In order to utilize the PyTorch model zoo, we use pytorch-style ResNet in our experiments.

In the meanwhile, we train models with caffe-style ResNet in 1x experiments for comparison.
We find that pytorch-style ResNet usually converges slower than caffe-style ResNet,
thus leading to slightly lower results in 1x schedule, but the final results
of 2x schedule is higher.

We report results using both caffe-style (weights converted from
[here](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#imagenet-pretrained-models))
and pytorch-style (weights from the official model zoo) ResNet backbone,
indicated as *pytorch-style results* / *caffe-style results*.

<table>
  <tr>
    <th>Type</th>
    <th>Lr schd</th>
    <th>Detectron</th>
    <th>Detectron.pytorch</th>
    <th>mmdetection</th>
  </tr>
  <tr>
    <td rowspan="2">RPN</td>
    <td>1x</td>
    <td>57.2</td>
    <td>-</td>
    <td>57.1 / 58.2</td>
  </tr>
  <tr>
    <td>2x</td>
    <td>-</td>
    <td>-</td>
    <td>57.6 / -</td>
  </tr>
  <tr>
    <td rowspan="2">Faster R-CNN</td>
    <td>1x</td>
    <td>36.7</td>
    <td>37.1</td>
    <td>36.4 / 36.7</td>
  </tr>
  <tr>
    <td>2x</td>
    <td>37.9</td>
    <td>-</td>
    <td>37.7 / -</td>
  </tr>
  <tr>
    <td rowspan="2">Mask R-CNN</td>
    <td>1x</td>
    <td>37.7 &amp; 33.9</td>
    <td>37.7 &amp; 33.7</td>
    <td>37.3 &amp; 34.2 / 37.5 &amp; 34.4</td>
  </tr>
  <tr>
    <td>2x</td>
    <td>38.6 &amp; 34.5</td>
    <td>-</td>
    <td>38.6 &amp; 35.1 / -</td>
  </tr>
  <tr>
    <td rowspan="2">Fast R-CNN</td>
    <td>1x</td>
    <td>36.4</td>
    <td>-</td>
    <td>35.8 / 36.6</td>
  </tr>
  <tr>
    <td>2x</td>
    <td>36.8</td>
    <td>-</td>
    <td>37.1 / -</td>
  </tr>
  <tr>
    <td rowspan="2">Fast R-CNN (w/mask)</td>
    <td>1x</td>
    <td>37.3 &amp; 33.7</td>
    <td>-</td>
    <td>36.8 &amp; 34.1 / 37.3 &amp; 34.5</td>
  </tr>
  <tr>
    <td>2x</td>
    <td>37.7 &amp; 34.0</td>
    <td>-</td>
    <td>37.9 &amp; 34.8 / -</td>
  </tr>
</table>

### Training Speed

The training speed is measure with s/iter. The lower, the better.

<table>
  <tr>
    <th>Type</th>
    <th>Detectron (P100<sup>1</sup>)</th>
    <th>Detectron.pytorch (XP<sup>2</sup>)</th>
    <th>mmdetection<sup>3</sup> (V100<sup>4</sup> / XP)</th>
  </tr>
  <tr>
    <td>RPN</td>
    <td>0.416</td>
    <td>-</td>
    <td>0.407 / 0.413</td>
  </tr>
  <tr>
    <td>Faster R-CNN</td>
    <td>0.544</td>
    <td>1.015</td>
    <td>0.554 / 0.579</td>
  </tr>
  <tr>
    <td>Mask R-CNN</td>
    <td>0.889</td>
    <td>1.435</td>
    <td>0.690 / 0.732</td>
  </tr>
  <tr>
    <td>Fast R-CNN</td>
    <td>0.285</td>
    <td>-</td>
    <td>0.375 / 0.398</td>
  </tr>
  <tr>
    <td>Fast R-CNN (w/mask)</td>
    <td>0.377</td>
    <td>-</td>
    <td>0.504 / 0.574</td>
  </tr>
</table>

\*1. Detectron reports the speed on Facebook's Big Basin servers (P100),
on our V100 servers it is slower so we use the official reported values.

\*2. Detectron.pytorch does not report the runtime and we encountered some issue to
run it on V100, so we report the speed on TITAN XP.

\*3. The speed of pytorch-style ResNet is approximately 5% slower than caffe-style,
and we report the pytorch-style results here.

\*4. We also run the models on a DGX-1 server (P100) and the speed is almost the same as our V100 servers.

### Inference Speed

The inference speed is measured with fps (img/s) on a single GPU. The higher, the better.

<table>
  <tr>
    <th>Type</th>
    <th>Detectron (P100)</th>
    <th>Detectron.pytorch (XP)</th>
    <th>mmdetection (V100 / XP)</th>
  </tr>
  <tr>
    <td>RPN</td>
    <td>12.5</td>
    <td>-</td>
    <td>14.5 / 15.4</td>
  </tr>
  <tr>
    <td>Faster R-CNN</td>
    <td>10.3</td>
    <td></td>
    <td>9.9 / 9.8</td>
  </tr>
  <tr>
    <td>Mask R-CNN</td>
    <td>8.5</td>
    <td></td>
    <td>7.7 / 7.4</td>
  </tr>
  <tr>
    <td>Fast R-CNN</td>
    <td>12.5</td>
    <td></td>
    <td>14.5 / 14.1</td>
  </tr>
  <tr>
    <td>Fast R-CNN (w/mask)</td>
    <td>9.9</td>
    <td></td>
    <td>10.6 / 10.3</td>
  </tr>
</table>

### Training memory

We perform various tests and there is no doubt that mmdetection is more memory
efficient than Detectron, and the main cause is the deep learning framework itself, not our efforts.
Besides, Caffe2 and PyTorch have different apis to obtain memory usage
whose implementation is not exactly the same.

`nvidia-smi` shows a larger memory usage for both detectron and mmdetection, e.g.,
we observe a much higher memory usage when we train Mask R-CNN with 2 images per GPU using detectron (10.6G) and mmdetection (9.3G), which is obviously more than actually required.

> With mmdetection, we can train R-50 FPN Mask R-CNN with **4** images per GPU (TITAN XP, 12G),
which is a promising result.
