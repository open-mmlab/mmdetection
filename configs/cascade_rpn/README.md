# Cascade RPN

> [Cascade RPN: Delving into High-Quality Region Proposal Network with Adaptive Convolution](https://arxiv.org/abs/1909.06720)

<!-- [ALGORITHM] -->

## Abstract

This paper considers an architecture referred to as Cascade Region Proposal Network (Cascade RPN) for improving the region-proposal quality and detection performance by systematically addressing the limitation of the conventional RPN that heuristically defines the anchors and aligns the features to the anchors. First, instead of using multiple anchors with predefined scales and aspect ratios, Cascade RPN relies on a single anchor per location and performs multi-stage refinement. Each stage is progressively more stringent in defining positive samples by starting out with an anchor-free metric followed by anchor-based metrics in the ensuing stages. Second, to attain alignment between the features and the anchors throughout the stages, adaptive convolution is proposed that takes the anchors in addition to the image features as its input and learns the sampled features guided by the anchors. A simple implementation of a two-stage Cascade RPN achieves AR 13.4 points higher than that of the conventional RPN, surpassing any existing region proposal methods. When adopting to Fast R-CNN and Faster R-CNN, Cascade RPN can improve the detection mAP by 3.1 and 3.5 points, respectively.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143872368-1580193a-d19c-4723-a579-c7ed2d5da4d1.png"/>
</div>

## Results and Models

### Region proposal performance

| Method | Backbone | Style | Mem (GB) | Train time (s/iter) | Inf time (fps) | AR 1000 |                                                      Config                                                       |                                                                    Download                                                                    |
| :----: | :------: | :---: | :------: | :-----------------: | :------------: | :-----: | :---------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
|  CRPN  | R-50-FPN | caffe |    -     |          -          |       -        |  72.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rpn/crpn_r50_caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/cascade_rpn/crpn_r50_caffe_fpn_1x_coco/cascade_rpn_r50_caffe_fpn_1x_coco-7aa93cef.pth) |

### Detection performance

|    Method    |  Proposal   | Backbone | Style | Schedule | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP |                                                            Config                                                             |                                                                            Download                                                                             |
| :----------: | :---------: | :------: | :---: | :------: | :------: | :-----------------: | :------------: | :----: | :---------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Fast R-CNN  | Cascade RPN | R-50-FPN | caffe |    1x    |    -     |          -          |       -        |  39.9  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rpn/crpn_fast_rcnn_r50_caffe_fpn_1x_coco.py)  |   [model](https://download.openmmlab.com/mmdetection/v2.0/cascade_rpn/crpn_fast_rcnn_r50_caffe_fpn_1x_coco/crpn_fast_rcnn_r50_caffe_fpn_1x_coco-cb486e66.pth)   |
| Faster R-CNN | Cascade RPN | R-50-FPN | caffe |    1x    |    -     |          -          |       -        |  40.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco/crpn_faster_rcnn_r50_caffe_fpn_1x_coco-c8283cca.pth) |

## Citation

We provide the code for reproducing experiment results of [Cascade RPN](https://arxiv.org/abs/1909.06720).

```latex
@inproceedings{vu2019cascade,
  title={Cascade RPN: Delving into High-Quality Region Proposal Network with Adaptive Convolution},
  author={Vu, Thang and Jang, Hyunjun and Pham, Trung X and Yoo, Chang D},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```
