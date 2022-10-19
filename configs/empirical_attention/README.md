# Empirical Attention

> [An Empirical Study of Spatial Attention Mechanisms in Deep Networks](https://arxiv.org/abs/1904.05873)

<!-- [ALGORITHM] -->

## Abstract

Attention mechanisms have become a popular component in deep neural networks, yet there has been little examination of how different influencing factors and methods for computing attention from these factors affect performance. Toward a better general understanding of attention mechanisms, we present an empirical study that ablates various spatial attention elements within a generalized attention formulation, encompassing the dominant Transformer attention as well as the prevalent deformable convolution and dynamic convolution modules. Conducted on a variety of applications, the study yields significant findings about spatial attention in deep networks, some of which run counter to conventional understanding. For example, we find that the query and key content comparison in Transformer attention is negligible for self-attention, but vital for encoder-decoder attention. A proper combination of deformable convolution with key content only saliency achieves the best accuracy-efficiency tradeoff in self-attention. Our results suggest that there exists much room for improvement in the design of attention mechanisms.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143879619-f1817da9-1573-45c9-891d-cfe55ad54911.png"/>
</div>

## Results and Models

| Backbone | Attention Component | DCN | Lr schd | Mem (GB) | Inf time (fps) | box AP |                         Config                          |                                                                                                                                                                                               Download                                                                                                                                                                                                |
| :------: | :-----------------: | :-: | :-----: | :------: | :------------: | :----: | :-----------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   |        1111         |  N  |   1x    |   8.0    |      13.8      |  40.0  |   [config](./faster-rcnn_r50-attn1111_fpn_1x_coco.py)   |         [model](https://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_1x_coco/faster_rcnn_r50_fpn_attention_1111_1x_coco_20200130-403cccba.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_1x_coco/faster_rcnn_r50_fpn_attention_1111_1x_coco_20200130_210344.log.json)         |
|   R-50   |        0010         |  N  |   1x    |   4.2    |      18.4      |  39.1  |   [config](./faster-rcnn_r50-attn0010_fpn_1x_coco.py)   |         [model](https://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_1x_coco/faster_rcnn_r50_fpn_attention_0010_1x_coco_20200130-7cb0c14d.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_1x_coco/faster_rcnn_r50_fpn_attention_0010_1x_coco_20200130_210125.log.json)         |
|   R-50   |        1111         |  Y  |   1x    |   8.0    |      12.7      |  42.1  | [config](./faster-rcnn_r50-attn1111-dcn_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco_20200130-8b2523a6.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco_20200130_204442.log.json) |
|   R-50   |        0010         |  Y  |   1x    |   4.2    |      17.1      |  42.0  | [config](./faster-rcnn_r50-attn0010-dcn_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco_20200130-1a2e831d.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco_20200130_210410.log.json) |

## Citation

```latex
@article{zhu2019empirical,
  title={An Empirical Study of Spatial Attention Mechanisms in Deep Networks},
  author={Zhu, Xizhou and Cheng, Dazhi and Zhang, Zheng and Lin, Stephen and Dai, Jifeng},
  journal={arXiv preprint arXiv:1904.05873},
  year={2019}
}
```
