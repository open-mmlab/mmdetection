# An Empirical Study of Spatial Attention Mechanisms in Deep Networks

## Introduction

```
@article{zhu2019empirical,
  title={An Empirical Study of Spatial Attention Mechanisms in Deep Networks},
  author={Zhu, Xizhou and Cheng, Dazhi and Zhang, Zheng and Lin, Stephen and Dai, Jifeng},
  journal={arXiv preprint arXiv:1904.05873},
  year={2019}
}
```


## Results and Models

| Backbone  | Attention Component | DCN  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
|:---------:|:-------------------:|:----:|:-------:|:--------:|:--------------:|:------:|:--------:|
| R-50      | 1111                | N    | 1x      | 8.0      | 5.6            | 40.0   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_1x_coco/faster_rcnn_r50_fpn_attention_1111_1x_coco_20200130-403cccba.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_1x_coco/faster_rcnn_r50_fpn_attention_1111_1x_coco_20200130_210344.log.json) |
| R-50      | 0010                | N    | 1x      | 4.2      | 17.0           | 39.1   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_1x_coco/faster_rcnn_r50_fpn_attention_0010_1x_coco_20200130-7cb0c14d.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_1x_coco/faster_rcnn_r50_fpn_attention_0010_1x_coco_20200130_210125.log.json) |
| R-50      | 1111                | Y    | 1x      | 8.0      | 5.4            | 42.1   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco_20200130-8b2523a6.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco_20200130_204442.log.json) |
| R-50      | 0010                | Y    | 1x      | 4.2      | 15.7           | 42.0   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco_20200130-1a2e831d.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco_20200130_210410.log.json) |
