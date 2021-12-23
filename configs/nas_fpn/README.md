# NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection

## Abstract

<!-- [ABSTRACT] -->

Current state-of-the-art convolutional architectures for object detection are manually designed. Here we aim to learn a better architecture of feature pyramid network for object detection. We adopt Neural Architecture Search and discover a new feature pyramid architecture in a novel scalable search space covering all cross-scale connections. The discovered architecture, named NAS-FPN, consists of a combination of top-down and bottom-up connections to fuse features across scales. NAS-FPN, combined with various backbone models in the RetinaNet framework, achieves better accuracy and latency tradeoff compared to state-of-the-art object detection models. NAS-FPN improves mobile detection accuracy by 2 AP compared to state-of-the-art SSDLite with MobileNetV2 model in [32] and achieves 48.3 AP which surpasses Mask R-CNN [10] detection accuracy with less computation time.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143968037-cedd76e9-1ae7-4869-bd34-c9d8611d630c.png"/>
</div>

<!-- [PAPER_TITLE: Nas-fpn: Learning scalable feature pyramid architecture for object detection] -->
<!-- [PAPER_URL: https://arxiv.org/abs/1904.07392] -->

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{ghiasi2019fpn,
  title={Nas-fpn: Learning scalable feature pyramid architecture for object detection},
  author={Ghiasi, Golnaz and Lin, Tsung-Yi and Le, Quoc V},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7036--7045},
  year={2019}
}
```

## Results and Models

We benchmark the new training schedule (crop training, large batch, unfrozen BN, 50 epochs) introduced in NAS-FPN. RetinaNet is used in the paper.

| Backbone    | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:-----------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50-FPN    | 50e     | 12.9     | 22.9           | 37.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/nas_fpn/retinanet_r50_fpn_crop640_50e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_fpn_crop640_50e_coco/retinanet_r50_fpn_crop640_50e_coco-9b953d76.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_fpn_crop640_50e_coco/retinanet_r50_fpn_crop640_50e_coco_20200529_095329.log.json) |
| R-50-NASFPN | 50e     | 13.2     | 23.0           | 40.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco/retinanet_r50_nasfpn_crop640_50e_coco_20200528_230008.log.json) |

**Note**: We find that it is unstable to train NAS-FPN and there is a small chance that results can be 3% mAP lower.
