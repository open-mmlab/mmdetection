# SOLOv2

> [SOLOv2: Dynamic and Fast Instance Segmentation](https://arxiv.org/abs/2003.10152)

<!-- [ALGORITHM] -->

## Abstract

In this work, we aim at building a simple, direct, and fast instance segmentation
framework with strong performance. We follow the principle of the SOLO method of
Wang et al. "SOLO: segmenting objects by locations". Importantly, we take one
step further by dynamically learning the mask head of the object segmenter such
that the mask head is conditioned on the location. Specifically, the mask branch
is decoupled into a mask kernel branch and mask feature branch, which are
responsible for learning the convolution kernel and the convolved features
respectively. Moreover, we propose Matrix NMS (non maximum suppression) to
significantly reduce the inference time overhead due to NMS of masks. Our
Matrix NMS performs NMS with parallel matrix operations in one shot, and
yields better results. We demonstrate a simple direct instance segmentation
system, outperforming a few state-of-the-art methods in both speed and accuracy.
A light-weight version of SOLOv2 executes at 31.3 FPS and yields 37.1% AP.
Moreover, our state-of-the-art results in object detection (from our mask byproduct)
and panoptic segmentation show the potential to serve as a new strong baseline
for many instance-level recognition tasks besides instance segmentation.

<div align=center>
<img src="https://user-images.githubusercontent.com/48282753/167235090-f20dab74-43a5-44ed-9f11-4e5f08866f45.png"/>
</div>

## Results and Models

### SOLOv2

|  Backbone  |  Style  | MS train | Lr schd | Mem (GB) | mask AP |                                                    Config                                                     |                                                                                                                                                Download                                                                                                                                                |
| :--------: | :-----: | :------: | :-----: | :------: | :-----: | :-----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50    | pytorch |    N     |   1x    |   5.1    |  34.8   |   [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_r50_fpn_1x_coco.py)    |      [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_1x_coco/solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth)           \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_1x_coco/solov2_r50_fpn_1x_coco_20220512_125858.log.json)      |
|    R-50    | pytorch |    Y     |   3x    |   5.1    |  37.5   |   [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_r50_fpn_3x_coco.py)    |      [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_3x_coco/solov2_r50_fpn_3x_coco_20220512_125856-fed092d4.pth)           \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_3x_coco/solov2_r50_fpn_3x_coco_20220512_125856.log.json)      |
|   R-101    | pytorch |    Y     |   3x    |   6.9    |  39.1   |   [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_r101_fpn_3x_coco.py)   |     [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r101_fpn_3x_coco/solov2_r101_fpn_3x_coco_20220511_095119-c559a076.pth)         \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r101_fpn_3x_coco/solov2_r101_fpn_3x_coco_20220511_095119.log.json)     |
| R-101(DCN) | pytorch |    Y     |   3x    |   7.1    |  41.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_r101_dcn_fpn_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r101_dcn_fpn_3x_coco/solov2_r101_dcn_fpn_3x_coco_20220513_214734-16c966cb.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r101_dcn_fpn_3x_coco/solov2_r101_dcn_fpn_3x_coco_20220513_214734.log.json) |
| X-101(DCN) | pytorch |    Y     |   3x    |   11.3   |  42.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_x101_dcn_fpn_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_x101_dcn_fpn_3x_coco/solov2_x101_dcn_fpn_3x_coco_20220513_214337-aef41095.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_x101_dcn_fpn_3x_coco/solov2_x101_dcn_fpn_3x_coco_20220513_214337.log.json) |

### Light SOLOv2

| Backbone |  Style  | MS train | Lr schd | Mem (GB) | mask AP |                                                     Config                                                     |                                                                                                                                                  Download                                                                                                                                                  |
| :------: | :-----: | :------: | :-----: | :------: | :-----: | :------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-18   | pytorch |    Y     |   3x    |   9.1    |  29.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_light_r18_fpn_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_light_r18_fpn_3x_coco/solov2_light_r18_fpn_3x_coco_20220511_083717-75fa355b.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_light_r18_fpn_3x_coco/solov2_light_r18_fpn_3x_coco_20220511_083717.log.json) |
|   R-34   | pytorch |    Y     |   3x    |   9.3    |  31.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_light_r34_fpn_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_light_r34_fpn_3x_coco/solov2_light_r34_fpn_3x_coco_20220511_091839-e51659d3.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_light_r34_fpn_3x_coco/solov2_light_r34_fpn_3x_coco_20220511_091839.log.json) |
|   R-50   | pytorch |    Y     |   3x    |   9.9    |  33.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_light_r50_fpn_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_light_r50_fpn_3x_coco/solov2_light_r50_fpn_3x_coco_20220512_165256-c93a6074.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_light_r50_fpn_3x_coco/solov2_light_r50_fpn_3x_coco_20220512_165256.log.json) |

## Citation

```latex
@article{wang2020solov2,
  title={SOLOv2: Dynamic and Fast Instance Segmentation},
  author={Wang, Xinlong and Zhang, Rufeng and  Kong, Tao and Li, Lei and Shen, Chunhua},
  journal={Proc. Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```
