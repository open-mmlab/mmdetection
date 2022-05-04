# NAS-FCOS

> [NAS-FCOS: Fast Neural Architecture Search for Object Detection](https://arxiv.org/abs/1906.04423)

<!-- [ALGORITHM] -->

## Abstract

The success of deep neural networks relies on significant architecture engineering. Recently neural architecture search (NAS) has emerged as a promise to greatly reduce manual effort in network design by automatically searching for optimal architectures, although typically such algorithms need an excessive amount of computational resources, e.g., a few thousand GPU-days. To date, on challenging vision tasks such as object detection, NAS, especially fast versions of NAS, is less studied. Here we propose to search for the decoder structure of object detectors with search efficiency being taken into consideration. To be more specific, we aim to efficiently search for the feature pyramid network (FPN) as well as the prediction head of a simple anchor-free object detector, namely FCOS, using a tailored reinforcement learning paradigm. With carefully designed search space, search algorithms and strategies for evaluating network quality, we are able to efficiently search a top-performing detection architecture within 4 days using 8 V100 GPUs. The discovered architecture surpasses state-of-the-art object detection models (such as Faster R-CNN, RetinaNet and FCOS) by 1.5 to 3.5 points in AP on the COCO dataset, with comparable computation complexity and memory footprint, demonstrating the efficacy of the proposed NAS for object detection.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143967900-1c8a65b9-c58d-4b03-8900-96af8f9768e8.png"/>
</div>

## Results and Models

| Head      | Backbone  | Style   | GN-head | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:---------:|:-------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| NAS-FCOSHead | R-50   | caffe   | Y       | 1x      |          |                | 39.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200520-1bdba3ce.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200520.log.json) |
| FCOSHead  | R-50      | caffe   | Y       | 1x      |          |                | 38.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200521-7fdcbce0.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200521.log.json) |

**Notes:**

- To be consistent with the author's implementation, we use 4 GPUs with 4 images/GPU.

## Citation

```latex
@article{wang2019fcos,
  title={Nas-fcos: Fast neural architecture search for object detection},
  author={Wang, Ning and Gao, Yang and Chen, Hao and Wang, Peng and Tian, Zhi and Shen, Chunhua},
  journal={arXiv preprint arXiv:1906.04423},
  year={2019}
}
```
