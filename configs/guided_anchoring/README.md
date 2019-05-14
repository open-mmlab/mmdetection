# Region Proposal by Guided Anchoring

## Introduction

We provide config files to reproduce the results in the CVPR 2019 paper for [Region Proposal by Guided Anchoring](https://arxiv.org/abs/1901.03278).

```
@inproceedings{wang2019region,
    title={Region Proposal by Guided Anchoring},
    author={Jiaqi Wang and Kai Chen and Shuo Yang and Chen Change Loy and Dahua Lin},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
    year={2019}
}
```

## Dataset

Guided Anchoring requires COCO dataset for training. You need to download and extract it in the COCO dataset path.
The directory should be like this.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

## Results and Models

The results on COCO 2017 val is shown in the below table. (results on test-dev are usually slightly higher than val).

|     Method     | Backbone | Style | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | AR 1000 | box AP |                                                                  Download                                                                  |
| :------------: | :------: | :---: | :-----: | :------: | :-----------------: | :------------: | :-----: | :----: | :----------------------------------------------------------------------------------------------------------------------------------------: |
|     GA-RPN     | R-50-FPN | caffe |   1x    |   5.0    |        0.55         |      13.3      |  68.5   |   -    |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/guided_anchoring/ga_rpn_r50_caffe_fpn_1x-95e91886.pth)    |
| GA-Faster RCNN | R-50-FPN | caffe |   1x    |   5.1    |        0.64         |      9.6       |    -    |  39.9  |  [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/guided_anchoring/ga_faster_r50_caffe_fpn_1x-a52b31fa.pth)   |
|  GA-Fast RCNN  | R-50-FPN | caffe |   1x    |   3.3    |        0.23         |      14.9      |    -    |  39.5  |   [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/guided_anchoring/ga_fast_r50_caffe_fpn_1x-c5af9f8b.pth)    |
|  GA-RetinaNet  | R-50-FPN | caffe |   1x    |   3.2    |        0.50         |      10.7      |    -    |  37.0  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x-29905101.pth) |


- In the Guided Anchoring paper, `score_thr` is set to 0.001 in Fast/Faster RCNN and 0.05 in RetinaNet for both baselines and Guided Anchoring .
- We use 8 Tesla V100 GPUs with 2 images/GPU for training.
- Inference time is evaluated on a single Tesla V100 GPU.