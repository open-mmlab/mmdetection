# Benchmark and Model Zoo

## Environment

### Hardware

- 8 NVIDIA Tesla V100 GPUs
- Intel Xeon 4114 CPU @ 2.20GHz

### Software environment

- Python 3.6 / 3.7
- PyTorch Nightly
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
|    R-101-FPN    | pytorch |   2x    |    - 