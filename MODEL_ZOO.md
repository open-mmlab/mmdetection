# Benchmark and Model Zoo

## Environment

### Hardware

- 8 NVIDIA Tesla V100 GPUs
- Intel Xeon 4114 CPU @ 2.20GHz

### Software environment

- Python 3.6 / 3.7
- PyTorch 1.1
- CUDA 9.0.176
- CUDNN 7.0.4
- NCCL 2.1.15

## Mirror sites

We use AWS as the main site to host our model zoo, and maintain a mirror on aliyun.
You can replace `https://s3.ap-northeast-2.amazonaws.com/open-mmlab` with `https://open-mmlab.oss-cn-beijing.aliyuncs.com` in model urls.

## Common settings

- All FPN baselines and RPN-C4 baselines were trained using 8 GPU with a batch size of 16 (2 images per GPU). Other C4 baselines were trained using 8 GPU with a batch size of 8 (1 image per GPU).
- All models were trained on `coco_2017_train`, and tested on the `coco_2017_val`.
- We use distributed training and BN layer stats are fixed.
- We adopt the same training schedules as Detectron. 1x indicates 12 epochs and 2x indicates 24 epochs, which corresponds to slightly less iterations than Detectron and the difference can be ignored.
- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo.
- For fair comparison with other codebases, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 8 GPUs. Note that this value is usually less than what `nvidia-smi` shows.
- We report the inference time as the overall time including data loading, network forwarding and post processing.


## Baselines

More models with different backbones will be added to the model zoo.

### RPN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | AR1000 |                                                          Download                                                          |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :------------------------------------------------------------------------------------------------------------------------: |
|     R-50-C4     |  caffe  |   1x    |    -     |          -          |      20.5      |  51.1  |      [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_caffe_c4_1x-ea7d3428.pth)       |
|     R-50-C4     |  caffe  |   2x    |   2.2    |        0.17         |      20.3      |  52.2  |      [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_caffe_c4_2x-c6d5b958.pth)       |
|     R-50-C4     | pytorch |   1x    |    -     |          -          |      20.1      |  50.2  |         [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_c4_1x-eb38972b.pth)          |
|     R-50-C4     | pytorch |   2x    |    -     |          -          |      20.0      |  51.1  |         [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_c4_2x-3d4c1e14.pth)          |
|    R-50-FPN     |  caffe  |   1x    |   3.3    |        0.253        |      16.9      |  58.2  |                                                             -                                                              |
|    R-50-FPN     | pytorch |   1x    |   3.5    |        0.276        |      17.7      |  57.1  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_fpn_1x_20181010-4a9c0712.pth)     |
|    R-50-FPN     | pytorch |   2x    |    -     |          -          |       -        |  57.6  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_fpn_2x_20181010-88a4a471.pth)     |
|    R-101-FPN    |  caffe  |   1x    |   5.2    |        0.379        |      13.9      |  59.4  |                                                             -                                                              |
|    R-101-FPN    | pytorch |   1x    |   5.4    |        0.396        |      14.4      |  58.6  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r101_fpn_1x_20181129-f50da4bd.pth)    |
|    R-101-FPN    | pytorch |   2x    |    -     |          -          |       -        |  59.1  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r101_fpn_2x_20181129-e42c6c9a.pth)    |
| X-101-32x4d-FPN | pytorch |   1x    |   6.6    |        0.589        |      11.8      |  59.4  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_x101_32x4d_fpn_1x_20181218-7e379d26.pth) |
| X-101-32x4d-FPN | pytorch |   2x    |    -     |          -          |       -        |  59.9  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_x101_32x4d_fpn_2x_20181218-0510af40.pth) |
| X-101-64x4d-FPN | pytorch |   1x    |   9.5    |        0.955        |      8.3       |  59.8  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_x101_64x4d_fpn_1x_20181218-c1a24f1f.pth) |
| X-101-64x4d-FPN | pytorch |   2x    |    -     |          -          |       -        |  60.0  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_x101_64x4d_fpn_2x_20181218-c22bdd70.pth) |

### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP |                                                              Download                                                              |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :--------------------------------------------------------------------------------------------------------------------------------: |
|     R-50-C4     |  caffe  |   1x    |    -     |          -          |      9.5       |  34.9  |      [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_caffe_c4_1x-75ecfdfa.pth)       |
|     R-50-C4     |  caffe  |   2x    |   4.0    |        0.39         |      9.3       |  36.5  |      [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_caffe_c4_2x-71c67f27.pth)       |
|     R-50-C4     | pytorch |   1x    |    -     |          -          |      9.3       |  33.9  |         [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_c4_1x-642cf91f.pth)          |
|     R-50-C4     | pytorch |   2x    |    -     |          -          |      9.4       |  35.9  |         [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_c4_2x-6e4fdf4f.pth)          |
|    R-50-FPN     |  caffe  |   1x    |   3.6    |        0.333        |      13.5      |  36.6  |                                                                 -                                                                  |
|    R-50-FPN     | pytorch |   1x    |   3.8    |        0.353        |      13.6      |  36.4  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth)     |
|    R-50-FPN     | pytorch |   2x    |    -     |          -          |       -        |  37.7  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth)     |
|    R-101-FPN    |  caffe  |   1x    |   5.5    |        0.465        |      11.5      |  38.8  |                                                                 -                                                                  |
|    R-101-FPN    | pytorch |   1x    |   5.7    |        0.474        |      11.9      |  38.5  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r101_fpn_1x_20181129-d1468807.pth)    |
|    R-101-FPN    | pytorch |   2x    |    -     |          -          |       -        |  39.4  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r101_fpn_2x_20181129-73e7ade7.pth)    |
| X-101-32x4d-FPN | pytorch |   1x    |   6.9    |        0.672        |      10.3      |  40.1  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_32x4d_fpn_1x_20181218-ad81c133.pth) |
| X-101-32x4d-FPN | pytorch |   2x    |    -     |          -          |       -        |  40.4  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_32x4d_fpn_2x_20181218-0ed58946.pth) |
| X-101-64x4d-FPN | pytorch |   1x    |   9.8    |        1.040        |      7.3       |  41.3  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_64x4d_fpn_1x_20181218-c9c69c8f.pth) |
| X-101-64x4d-FPN | pytorch |   2x    |    -     |          -          |       -        |  40.7  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_64x4d_fpn_2x_20181218-fe94f9b8.pth) |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP |                                                             Download                                                             |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :-----: | :------------------------------------------------------------------------------------------------------------------------------: |
|     R-50-C4     |  caffe  |   1x    |    -     |          -          |      8.1       |  35.9  |  31.5   |      [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_caffe_c4_1x-02a4ad3b.pth)       |
|     R-50-C4     |  caffe  |   2x    |   4.2    |        0.43         |      8.1       |  37.9  |  32.9   |      [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_caffe_c4_2x-d150973a.pth)       |
|     R-50-C4     | pytorch |   1x    |    -     |          -          |      7.9       |  35.1  |  31.2   |         [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_c4_1x-a83bdd40.pth)          |
|     R-50-C4     | pytorch |   2x    |    -     |          -          |      8.0       |  37.2  |  32.5   |         [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_c4_2x-3cf169a9.pth)          |
|    R-50-FPN     |  caffe  |   1x    |   3.8    |        0.430        |      10.2      |  37.4  |  34.3   |                                                                -                                                                 |
|    R-50-FPN     | pytorch |   1x    |   3.9    |        0.453        |      10.6      |  37.3  |  34.2   |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth)     |
|    R-50-FPN     | pytorch |   2x    |    -     |          -          |       -        |  38.5  |  35.1   |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth)     |
|    R-101-FPN    |  caffe  |   1x    |   5.7    |        0.534        |      9.4       |  39.9  |  36.1   |                                                                -                                                                 |
|    R-101-FPN    | pytorch |   1x    |   5.8    |        0.571        |      9.5       |  39.4  |  35.9   |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r101_fpn_1x_20181129-34ad1961.pth)    |
|    R-101-FPN    | pytorch |   2x    |    -     |          -          |       -        |  40.3  |  36.5   |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r101_fpn_2x_20181129-a254bdfc.pth)    |
| X-101-32x4d-FPN | pytorch |   1x    |   7.1    |        0.759        |      8.3       |  41.1  |  37.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_x101_32x4d_fpn_1x_20181218-44e635cc.pth) |
| X-101-32x4d-FPN | pytorch |   2x    |    -     |          -          |       -        |  41.4  |  37.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_x101_32x4d_fpn_2x_20181218-f023dffa.pth) |
| X-101-64x4d-FPN | pytorch |   1x    |   10.0   |        1.102        |      6.5       |  42.1  |  38.0   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_x101_64x4d_fpn_1x_20181218-cb159987.pth) |
| X-101-64x4d-FPN | pytorch |   2x    |    -     |          -          |       -        |  42.0  |  37.7   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_x101_64x4d_fpn_2x_20181218-ea936e44.pth) |

### Fast R-CNN (with pre-computed proposals)

| Backbone  |  Style  |  Type  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP |                                                            Download                                                             |
| :-------: | :-----: | :----: | :-----: | :------: | :-----------------: | :------------: | :----: | :-----: | :-----------------------------------------------------------------------------------------------------------------------------: |
|  R-50-C4  |  caffe  | Faster |   1x    |    -     |          -          |      6.7       |  35.0  |    -    |      [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_rcnn_r50_caffe_c4_1x-0ef9a60b.pth)      |
|  R-50-C4  |  caffe  | Faster |   2x    |   3.8    |        0.34         |      6.6       |  36.4  |    -    |         [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_rcnn_r50_c4_2x-657a9fc6.pth)         |
|  R-50-C4  | pytorch | Faster |   1x    |    -     |          -          |      6.3       |  34.2  |    -    |         [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_rcnn_r50_c4_1x-2bc00ca9.pth)         |
|  R-50-C4  | pytorch | Faster |   2x    |    -     |          -          |      6.1       |  35.8  |    -    |      [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_rcnn_r50_caffe_c4_2x-9171d0fc.pth)      |
| R-50-FPN  |  caffe  | Faster |   1x    |   3.3    |        0.242        |      18.4      |  36.6  |    -    |                                                                -                                                                |
| R-50-FPN  | pytorch | Faster |   1x    |   3.5    |        0.250        |      16.5      |  35.8  |    -    |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_rcnn_r50_fpn_1x_20181010-08160859.pth)    |
|  R-50-C4  |  caffe  |  Mask  |   1x    |    -     |          -          |      8.1       |  35.9  |  31.5   |   [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_mask_rcnn_r50_caffe_c4_1x-b43f7f3c.pth)    |
|  R-50-C4  |  caffe  |  Mask  |   2x    |   4.2    |        0.43         |      8.1       |  37.9  |  32.9   |   [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_mask_rcnn_r50_caffe_c4_2x-e3580184.pth)    |
|  R-50-C4  | pytorch |  Mask  |   1x    |    -     |          -          |      7.9       |  35.1  |  31.2   |      [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_mask_rcnn_r50_c4_1x-bc7fa8c8.pth)       |
|  R-50-C4  | pytorch |  Mask  |   2x    |    -     |          -          |      8.0       |  37.2  |  32.5   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_mask_rcnn_r50_fpn_2x_20181010-5048cb03.pth)  |
| R-50-FPN  | pytorch | Faster |   2x    |    -     |          -          |       -        |  37.1  |    -    |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_rcnn_r50_fpn_2x_20181010-d263ada5.pth)    |
| R-101-FPN |  caffe  | Faster |   1x    |   5.2    |        0.355        |      14.4      |  38.6  |    -    |                                                                -                                                                |
| R-101-FPN | pytorch | Faster |   1x    |   5.4    |        0.388        |      13.2      |  38.1  |    -    |   [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_rcnn_r101_fpn_1x_20181129-ffaa2eb0.pth)    |
| R-101-FPN | pytorch | Faster |   2x    |    -     |          -          |       -        |  38.8  |    -    |   [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_rcnn_r101_fpn_2x_20181129-9dba92ce.pth)    |
| R-50-FPN  |  caffe  |  Mask  |   1x    |   3.4    |        0.328        |      12.8      |  37.3  |  34.5   |                                                                -                                                                |
| R-50-FPN  | pytorch |  Mask  |   1x    |   3.5    |        0.346        |      12.7      |  36.8  |  34.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_mask_rcnn_r50_fpn_1x_20181010-e030a38f.pth)  |
| R-50-FPN  | pytorch |  Mask  |   2x    |    -     |          -          |       -        |  37.9  |  34.8   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_mask_rcnn_r50_fpn_2x_20181010-5048cb03.pth)  |
| R-101-FPN |  caffe  |  Mask  |   1x    |   5.2    |        0.429        |      11.2      |  39.4  |  36.1   |                                                                -                                                                |
| R-101-FPN | pytorch |  Mask  |   1x    |   5.4    |        0.462        |      10.9      |  38.9  |  35.8   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_mask_rcnn_r101_fpn_1x_20181129-2273fa9b.pth) |
| R-101-FPN | pytorch |  Mask  |   2x    |    -     |          -          |       -        |  39.9  |  36.4   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_mask_rcnn_r101_fpn_2x_20181129-bf63ec5e.pth) |

### RetinaNet

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP |                                                             Download                                                             |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :------------------------------------------------------------------------------------------------------------------------------: |
|    R-50-FPN     |  caffe  |   1x    |   3.4    |        0.285        |      12.5      |  35.8  |                                                                -                                                                 |
|    R-50-FPN     | pytorch |   1x    |   3.6    |        0.308        |      12.1      |  35.6  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-7b0c2548.pth)     |
|    R-50-FPN     | pytorch |   2x    |    -     |          -          |       -        |  36.4  |    [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/retinanet_r50_fpn_2x_20190616-75574209.pth)     |
|    R-101-FPN    |  caffe  |   1x    |   5.3    |        0.410        |      10.4      |  37.8  |                                                                -                                                                 |
|    R-101-FPN    | pytorch |   1x    |   5.5    |        0.429        |      10.9      |  37.7  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r101_fpn_1x_20181129-f016f384.pth)    |
|    R-101-FPN    | pytorch |   2x    |    -     |          -          |       -        |  38.1  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r101_fpn_2x_20181129-72c14526.pth)    |
| X-101-32x4d-FPN | pytorch |   1x    |   6.7    |        0.632        |      9.3       |  39.0  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_32x4d_fpn_1x_20190501-967812ba.pth) |
| X-101-32x4d-FPN | pytorch |   2x    |    -     |          -          |       -        |  39.3  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_32x4d_fpn_2x_20181218-8596452d.pth) |
| X-101-64x4d-FPN | pytorch |   1x    |   9.6    |        0.993        |      7.0       |  40.0  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_64x4d_fpn_1x_20181218-a0a22662.pth) |
| X-101-64x4d-FPN | pytorch |   2x    |    -     |          -          |       -        |  39.6  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_64x4d_fpn_2x_20181218-5e88d045.pth) |

### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP |                                                              Download                                                               |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :---------------------------------------------------------------------------------------------------------------------------------: |
|     R-50-C4     |  caffe  |   1x    |   8.7    |        0.92         |      5.0       |  38.7  |      [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r50_caffe_c4_1x-7c85c62b.pth)       |
|    R-50-FPN     |  caffe  |   1x    |   3.9    |        0.464        |      10.9      |  40.5  |                                                                  -                                                                  |
|    R-50-FPN     | pytorch |   1x    |   4.1    |        0.455        |      11.9      |  40.4  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth)     |
|    R-50-FPN     | pytorch |   20e   |    -     |          -          |       -        |  41.1  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth)    |
|    R-101-FPN    |  caffe  |   1x    |   5.8    |        0.569        |      9.6       |  42.4  |                                                                  -                                                                  |
|    R-101-FPN    | pytorch |   1x    |   6.0    |        0.584        |      10.3      |  42.0  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth)    |
|    R-101-FPN    | pytorch |   20e   |    -     |          -          |       -        |  42.5  |   [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r101_fpn_20e_20181129-b46dcede.pth)    |
| X-101-32x4d-FPN | pytorch |   1x    |   7.2    |        0.770        |      8.9       |  43.6  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_32x4d_fpn_1x_20190501-af628be5.pth) |
| X-101-32x4d-FPN | pytorch |   20e   |    -     |          -          |       -        |  44.0  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_32x4d_fpn_2x_20181218-28f73c4c.pth) |
| X-101-64x4d-FPN | pytorch |   1x    |   10.0   |        1.133        |      6.7       |  44.5  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth) |
| X-101-64x4d-FPN | pytorch |   20e   |    -     |          -          |       -        |  44.7  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_64x4d_fpn_2x_20181218-5add321e.pth) |

### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP |                                                                 Download                                                                  |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :-----: | :---------------------------------------------------------------------------------------------------------------------------------------: |
|     R-50-C4     |  caffe  |   1x    |   9.1    |        0.99         |      4.5       |  39.3  |  32.8   |       [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r50_caffe_c4_1x-f72cc254.pth)       |
|    R-50-FPN     |  caffe  |   1x    |   5.1    |        0.692        |      7.6       |  40.9  |  35.5   |                                                                     -                                                                     |
|    R-50-FPN     | pytorch |   1x    |   5.3    |        0.683        |      7.4       |  41.2  |  35.7   |     [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r50_fpn_1x_20181123-88b170c9.pth)     |
|    R-50-FPN     | pytorch |   20e   |    -     |          -          |       -        |  42.3  |  36.6   |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r50_fpn_20e_20181123-6e0c9713.pth)     |
|    R-101-FPN    |  caffe  |   1x    |   7.0    |        0.803        |      7.2       |  43.1  |  37.2   |                                                                     -                                                                     |
|    R-101-FPN    | pytorch |   1x    |   7.2    |        0.807        |      6.8       |  42.6  |  37.0   |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r101_fpn_1x_20181129-64f00602.pth)     |
|    R-101-FPN    | pytorch |   20e   |    -     |          -          |       -        |  43.3  |  37.6   |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r101_fpn_20e_20181129-cb85151d.pth)    |
| X-101-32x4d-FPN | pytorch |   1x    |   8.4    |        0.976        |      6.6       |  44.4  |  38.2   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_x101_32x4d_fpn_1x_20181218-1d944c89.pth)  |
| X-101-32x4d-FPN | pytorch |   20e   |    -     |          -          |       -        |  44.7  |  38.6   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_x101_32x4d_fpn_20e_20181218-761a3473.pth) |
| X-101-64x4d-FPN | pytorch |   1x    |   11.4   |        1.33         |      5.3       |  45.4  |  39.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_x101_64x4d_fpn_1x_20190501-827e0a70.pth)  |
| X-101-64x4d-FPN | pytorch |   20e   |    -     |          -          |       -        |  45.7  |  39.4   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_x101_64x4d_fpn_20e_20181218-630773a7.pth) |

**Notes:**

- The `20e` schedule in Cascade (Mask) R-CNN indicates decreasing the lr at 16 and 19 epochs, with a total of 20 epochs.

### Hybrid Task Cascade (HTC)

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP |                                                            Download                                                             |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :-----: | :-----------------------------------------------------------------------------------------------------------------------------: |
|    R-50-FPN     | pytorch |   1x    |   7.4    |        0.936        |      4.1       |  42.1  |  37.3   |     [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_r50_fpn_1x_20190408-878c1712.pth)     |
|    R-50-FPN     | pytorch |   20e   |    -     |          -          |       -        |  43.2  |  38.1   |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_r50_fpn_20e_20190408-c03b7015.pth)     |
|    R-101-FPN    | pytorch |   20e   |   9.3    |        1.051        |      4.0       |  44.9  |  39.4   |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_r101_fpn_20e_20190408-a2e586db.pth)    |
| X-101-32x4d-FPN | pytorch |   20e   |   5.8    |        0.769        |      3.8       |  46.1  |  40.3   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_x101_32x4d_fpn_20e_20190408-9eae4d0b.pth) |
| X-101-64x4d-FPN | pytorch |   20e   |   7.5    |        1.120        |      3.5       |  46.9  |  40.8   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_x101_64x4d_fpn_20e_20190408-497f2561.pth) |

**Notes:**

- Please refer to [Hybrid Task Cascade](configs/htc/README.md) for details and more a powerful model (50.7/43.9).

### SSD

| Backbone | Size  | Style | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP |                                                             Download                                                              |
| :------: | :---: | :---: | :-----: | :------: | :-----------------: | :------------: | :----: | :-------------------------------------------------------------------------------------------------------------------------------: |
|  VGG16   |  300  | caffe |  120e   |   3.5    |        0.256        |  25.9 / 34.6   |  25.7  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth) |
|  VGG16   |  512  | caffe |  120e   |   7.6    |        0.412        |  20.7 / 25.4   |  29.3  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ssd512_coco_vgg16_caffe_120e_20181221-d48b0be8.pth) |

**Notes:**

- `cudnn.benchmark` is set as `True` for SSD training and testing.
- Inference time is reported for batch size = 1 and batch size = 8.
- The speed on COCO and VOC are different due to model parameters and nms.

### Group Normalization (GN)

Please refer to [Group Normalization](configs/gn/README.md) for details.

### Weight Standardization

Please refer to [Weight Standardization](configs/gn+ws/README.md) for details.

### Deformable Convolution v2

Please refer to [Deformable Convolutional Networks](configs/dcn/README.md) for details.

### Libra R-CNN

Please refer to [Libra R-CNN](configs/libra_rcnn/README.md) for details.

### Guided Anchoring

Please refer to [Guided Anchoring](configs/guided_anchoring/README.md) for details.

### FCOS

Please refer to [FCOS](configs/fcos/README.md) for details.

### Grid R-CNN (plus)

Please refer to [Grid R-CNN](configs/grid_rcnn/README.md) for details.

### GHM

Please refer to [GHM](configs/ghm/README.md) for details.

### GCNet

Please refer to [GCNet](configs/gcnet/README.md) for details.

### HRNet
Please refer to [HRNet](configs/hrnet/README.md) for details.

### Mask Scoring R-CNN

Please refer to [Mask Scoring R-CNN](configs/ms_rcnn/README.md) for details.

### Train from Scratch

Please refer to [Rethinking ImageNet Pre-training](configs/scratch/README.md) for details.

### Other datasets

We also benchmark some methods on [PASCAL VOC](configs/pascal_voc/README.md), [Cityscapes](configs/cityscapes/README.md) and [WIDER FACE](configs/wider_face/README.md).


## Comparison with Detectron and maskrcnn-benchmark

We compare mmdetection with [Detectron](https://github.com/facebookresearch/Detectron)
and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). The backbone used is R-50-FPN.

In general, mmdetection has 3 advantages over Detectron.

- **Higher performance** (especially in terms of mask AP)
- **Faster training speed**
- **Memory efficient**

### Performance

Detectron and maskrcnn-benchmark use caffe-style ResNet as the backbone.
We report results using both caffe-style (weights converted from
[here](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#imagenet-pretrained-models))
and pytorch-style (weights from the official model zoo) ResNet backbone,
indicated as *pytorch-style results* / *caffe-style results*.

We find that pytorch-style ResNet usually converges slower than caffe-style ResNet,
thus leading to slightly lower results in 1x schedule, but the final results
of 2x schedule is higher.

<table>
  <tr>
    <th>Type</th>
    <th>Lr schd</th>
    <th>Detectron</th>
    <th>maskrcnn-benchmark</th>
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
    <td>36.8</td>
    <td>36.4 / 36.6</td>
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
    <td>37.8 &amp; 34.2</td>
    <td>37.3 &amp; 34.2 / 37.4 &amp; 34.3</td>
  </tr>
  <tr>
    <td>2x</td>
    <td>38.6 &amp; 34.5</td>
    <td>-</td>
    <td>38.5 &amp; 35.1 / -</td>
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
    <th>maskrcnn-benchmark (V100)</th>
    <th>mmdetection (V100<sup>2</sup>)</th>
  </tr>
  <tr>
    <td>RPN</td>
    <td>0.416</td>
    <td>-</td>
    <td>0.253</td>
  </tr>
  <tr>
    <td>Faster R-CNN</td>
    <td>0.544</td>
    <td>0.353</td>
    <td>0.333</td>
  </tr>
  <tr>
    <td>Mask R-CNN</td>
    <td>0.889</td>
    <td>0.454</td>
    <td>0.430</td>
  </tr>
  <tr>
    <td>Fast R-CNN</td>
    <td>0.285</td>
    <td>-</td>
    <td>0.242</td>
  </tr>
  <tr>
    <td>Fast R-CNN (w/mask)</td>
    <td>0.377</td>
    <td>-</td>
    <td>0.328</td>
  </tr>
</table>

\*1. Facebook's Big Basin servers (P100/V100) is slightly faster than the servers we use. mmdetection can also run slightly faster on FB's servers.

\*2. For fair comparison, we list the caffe-style results here.


### Inference Speed

The inference speed is measured with fps (img/s) on a single GPU. The higher, the better.

<table>
  <tr>
    <th>Type</th>
    <th>Detectron (P100)</th>
    <th>maskrcnn-benchmark (V100)</th>
    <th>mmdetection (V100)</th>
  </tr>
  <tr>
    <td>RPN</td>
    <td>12.5</td>
    <td>-</td>
    <td>16.9</td>
  </tr>
  <tr>
    <td>Faster R-CNN</td>
    <td>10.3</td>
    <td>7.9</td>
    <td>13.5</td>
  </tr>
  <tr>
    <td>Mask R-CNN</td>
    <td>8.5</td>
    <td>7.7</td>
    <td>10.2</td>
  </tr>
  <tr>
    <td>Fast R-CNN</td>
    <td>12.5</td>
    <td>-</td>
    <td>18.4</td>
  </tr>
  <tr>
    <td>Fast R-CNN (w/mask)</td>
    <td>9.9</td>
    <td>-</td>
    <td>12.8</td>
  </tr>
</table>

### Training memory

<table>
  <tr>
    <th>Type</th>
    <th>Detectron</th>
    <th>maskrcnn-benchmark</th>
    <th>mmdetection</th>
  </tr>
  <tr>
    <td>RPN</td>
    <td>6.4</td>
    <td>-</td>
    <td>3.3</td>
  </tr>
  <tr>
    <td>Faster R-CNN</td>
    <td>7.2</td>
    <td>4.4</td>
    <td>3.6</td>
  </tr>
  <tr>
    <td>Mask R-CNN</td>
    <td>8.6</td>
    <td>5.2</td>
    <td>3.8</td>
  </tr>
  <tr>
    <td>Fast R-CNN</td>
    <td>6.0</td>
    <td>-</td>
    <td>3.3</td>
  </tr>
  <tr>
    <td>Fast R-CNN (w/mask)</td>
    <td>7.9</td>
    <td>-</td>
    <td>3.4</td>
  </tr>
</table>

There is no doubt that maskrcnn-benchmark and mmdetection is more memory efficient than Detectron,
and the main advantage is PyTorch itself. We also perform some memory optimizations to push it forward.

Note that Caffe2 and PyTorch have different apis to obtain memory usage with different implementations.
For all codebases, `nvidia-smi` shows a larger memory usage than the reported number in the above table.
