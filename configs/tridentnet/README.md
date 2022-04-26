# TridentNet

> [Scale-Aware Trident Networks for Object Detection](https://arxiv.org/abs/1901.01892)

<!-- [ALGORITHM] -->

## Abstract

Scale variation is one of the key challenges in object detection. In this work, we first present a controlled experiment to investigate the effect of receptive fields for scale variation in object detection. Based on the findings from the exploration experiments, we propose a novel Trident Network (TridentNet) aiming to generate scale-specific feature maps with a uniform representational power. We construct a parallel multi-branch architecture in which each branch shares the same transformation parameters but with different receptive fields. Then, we adopt a scale-aware training scheme to specialize each branch by sampling object instances of proper scales for training. As a bonus, a fast approximation version of TridentNet could achieve significant improvements without any additional parameters and computational cost compared with the vanilla detector. On the COCO dataset, our TridentNet with ResNet-101 backbone achieves state-of-the-art single-model results of 48.4 mAP.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143999668-0927922e-efc2-45fa-8bfc-1e3df18720f5.png"/>
</div>

## Results and Models

We reports the test results using only one branch for inference.

|    Backbone     |  Style  | mstrain | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
| :-------------: | :-----: | :-----: | :-----: | :------: | :------------: | :----: | :------: |
|    R-50         |  caffe  |    N    |   1x    |          |                | 37.7   |[model](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_1x_coco/tridentnet_r50_caffe_1x_coco_20201230_141838-2ec0b530.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_1x_coco/tridentnet_r50_caffe_1x_coco_20201230_141838.log.json) |
|    R-50         |  caffe  |    Y    |   1x    |          |                | 37.6   |[model](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_mstrain_1x_coco/tridentnet_r50_caffe_mstrain_1x_coco_20201230_141839-6ce55ccb.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_mstrain_1x_coco/tridentnet_r50_caffe_mstrain_1x_coco_20201230_141839.log.json) |
|    R-50         |  caffe  |    Y    |   3x    |          |                | 40.3   |[model](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_mstrain_3x_coco/tridentnet_r50_caffe_mstrain_3x_coco_20201130_100539-46d227ba.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_mstrain_3x_coco/tridentnet_r50_caffe_mstrain_3x_coco_20201130_100539.log.json) |

**Note**

Similar to [Detectron2](https://github.com/facebookresearch/detectron2/tree/master/projects/TridentNet), we haven't implemented the Scale-aware Training Scheme in section 4.2 of the paper.

## Citation

```latex
@InProceedings{li2019scale,
  title={Scale-Aware Trident Networks for Object Detection},
  author={Li, Yanghao and Chen, Yuntao and Wang, Naiyan and Zhang, Zhaoxiang},
  journal={The International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
