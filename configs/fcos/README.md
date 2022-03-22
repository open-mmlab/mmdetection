# FCOS

> [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)

<!-- [ALGORITHM] -->

## Abstract

We propose a fully convolutional one-stage object detector (FCOS) to solve object detection in a per-pixel prediction fashion, analogue to semantic segmentation. Almost all state-of-the-art object detectors such as RetinaNet, SSD, YOLOv3, and Faster R-CNN rely on pre-defined anchor boxes. In contrast, our proposed detector FCOS is anchor box free, as well as proposal free. By eliminating the predefined set of anchor boxes, FCOS completely avoids the complicated computation related to anchor boxes such as calculating overlapping during training. More importantly, we also avoid all hyper-parameters related to anchor boxes, which are often very sensitive to the final detection performance. With the only post-processing non-maximum suppression (NMS), FCOS with ResNeXt-64x4d-101 achieves 44.7% in AP with single-model and single-scale testing, surpassing previous one-stage detectors with the advantage of being much simpler. For the first time, we demonstrate a much simpler and flexible detection framework achieving improved detection accuracy. We hope that the proposed FCOS framework can serve as a simple and strong alternative for many other instance-level tasks.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143882011-45b234bc-d04b-4bbe-a822-94bec057ac86.png"/>
</div>

## Results and Models

| Backbone  | Style   | GN      | MS train | Tricks  | DCN     | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50      | caffe   | Y       | N        | N       | N       | 1x      | 3.6      | 22.7           | 36.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/20201227_180009.log.json) |
| R-50      | caffe   | Y       | N        | Y       | N       | 1x      | 3.7      | -              | 38.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/20210105_135818.log.json)|
| R-50      | caffe   | Y       | N        | Y       | Y       | 1x      | 3.8      | -              | 42.3   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco/20210105_224556.log.json)|
| R-101     | caffe   | Y       | N        | N       | N       | 1x      | 5.5      | 17.3           | 39.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r101_caffe_fpn_gn-head_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_1x_coco/fcos_r101_caffe_fpn_gn-head_1x_coco-0e37b982.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_1x_coco/20210103_155046.log.json) |

| Backbone  | Style   | GN      | MS train | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50      | caffe   | Y       | Y        | 2x      | 2.6      | 22.9           | 38.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco-d92ceeea.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco/20201227_161900.log.json) |
| R-101     | caffe   | Y       | Y        | 2x      | 5.5      | 17.3           | 40.8   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco/20210103_155046.log.json) |
| X-101     | pytorch | Y       | Y        | 2x      | 10.0     | 9.7            | 42.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco/20210114_133041.log.json) |

**Notes:**

- The X-101 backbone is X-101-64x4d.
- Tricks means setting `norm_on_bbox`, `centerness_on_reg`, `center_sampling` as `True`.
- DCN means using `DCNv2` in both backbone and head.

## Citation

```latex
@article{tian2019fcos,
  title={FCOS: Fully Convolutional One-Stage Object Detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal={arXiv preprint arXiv:1904.01355},
  year={2019}
}
```
