# DyHead

> [Dynamic Head: Unifying Object Detection Heads with Attentions](https://arxiv.org/abs/2106.08322)

<!-- [ALGORITHM] -->

## Abstract

The complex nature of combining localization and classification in object detection has resulted in the flourished development of methods. Previous works tried to improve the performance in various object detection heads but failed to present a unified view. In this paper, we present a novel dynamic head framework to unify object detection heads with attentions. By coherently combining multiple self-attention mechanisms between feature levels for scale-awareness, among spatial locations for spatial-awareness, and within output channels for task-awareness, the proposed approach significantly improves the representation ability of object detection heads without any computational overhead. Further experiments demonstrate that the effectiveness and efficiency of the proposed dynamic head on the COCO benchmark. With a standard ResNeXt-101-DCN backbone, we largely improve the performance over popular object detectors and achieve a new state-of-the-art at 54.0 AP. Furthermore, with latest transformer backbone and extra data, we can push current best COCO result to a new record at 60.6 AP.

<div align=center>
<img src="https://user-images.githubusercontent.com/42844407/149169448-fcafb6d0-b866-41cc-9422-94de9f1e1761.png" height="300"/>
</div>

## Results and Models

| Method | Backbone | Style   | Setting      | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:------:|:--------:|:-------:|:------------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| ATSS   | R-50     | caffe   | reproduction | 1x      | 5.4      | 13.2           | 42.5   | [config](./atss_r50_caffe_fpn_dyhead_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dyhead/atss_r50_fpn_dyhead_for_reproduction_1x_coco/atss_r50_fpn_dyhead_for_reproduction_4x4_1x_coco_20220107_213939-162888e6.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dyhead/atss_r50_fpn_dyhead_for_reproduction_1x_coco/atss_r50_fpn_dyhead_for_reproduction_4x4_1x_coco_20220107_213939.log.json) |
| ATSS   | R-50     | pytorch | simple       | 1x      | 4.9      | 13.7           | 43.3   | [config](./atss_r50_fpn_dyhead_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dyhead/atss_r50_fpn_dyhead_4x4_1x_coco/atss_r50_fpn_dyhead_4x4_1x_coco_20211219_023314-eaa620c6.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dyhead/atss_r50_fpn_dyhead_4x4_1x_coco/atss_r50_fpn_dyhead_4x4_1x_coco_20211219_023314.log.json) |

- We trained the above models with 4 GPUs and 4 `samples_per_gpu`.
- The `reproduction` setting aims to reproduce the official implementation based on Detectron2.
- The `simple` setting serves as a minimum example to use DyHead in MMDetection. Specifically,
  - it adds `DyHead` to `neck` after `FPN`
  - it sets `stacked_convs=0` to `bbox_head`
- The `simple` setting achieves higher AP than the original implementation.
  We have not conduct ablation study between the two settings.
  `dict(type='Pad', size_divisor=128)` may further improve AP by prefer spatial alignment across pyramid levels, although large padding reduces efficiency.

## Relation to Other Methods

- DyHead can be regarded as an improved [SEPC](https://arxiv.org/abs/2005.03101) with [DyReLU modules](https://arxiv.org/abs/2003.10027) and simplified [SE blocks](https://arxiv.org/abs/1709.01507).
- Xiyang Dai et al., the author team of DyHead, adopt it for [Dynamic DETR](https://openaccess.thecvf.com/content/ICCV2021/html/Dai_Dynamic_DETR_End-to-End_Object_Detection_With_Dynamic_Attention_ICCV_2021_paper.html).
  The description of Dynamic Encoder in Sec. 3.2 will help you understand DyHead.

## Citation

```latex
@inproceedings{DyHead_CVPR2021,
  author    = {Dai, Xiyang and Chen, Yinpeng and Xiao, Bin and Chen, Dongdong and Liu, Mengchen and Yuan, Lu and Zhang, Lei},
  title     = {Dynamic Head: Unifying Object Detection Heads With Attentions},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021}
}
```
