# RTMDet

<!-- [ALGORITHM] -->

## Abstract

Our tech-report will be released soon.

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/192182907-f9a671d6-89cb-4d73-abd8-c2b9dada3c66.png"/>
</div>

## Results and Models

|  Backbone   | size | box AP | Params(M) | FLOPS(G) | TRT-FP16-Latency(ms) |                   Config                   |                                                                                                                                              Download                                                                                                                                               |
| :---------: | :--: | :----: | :-------: | :------: | :------------------: | :----------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| RTMDet-tiny | 640  |  40.9  |    4.8    |   8.1    |         0.98         | [config](./rtmdet_tiny_8xb32-300e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220902_112414.log.json) |
|  RTMDet-s   | 640  |  44.5  |   8.89    |   14.8   |         1.22         |  [config](./rtmdet_s_8xb32-300e_coco.py)   |     [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-2724601d.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602.log.json)      |
|  RTMDet-m   | 640  |  49.1  |   24.71   |  39.27   |         1.62         |  [config](./rtmdet_m_8xb32-300e_coco.py)   |     [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220.log.json)      |
|  RTMDet-l   | 640  |  51.3  |   52.3    |  80.23   |         2.44         |  [config](./rtmdet_l_8xb32-300e_coco.py)   |     [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030.log.json)      |
|  RTMDet-x   | 640  |  52.6  |   94.86   |  141.67  |         3.10         |  [config](./rtmdet_x_8xb32-300e_coco.py)   |     [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555.log.json)      |

**Note**:

1. The inference speed is measured on an NVIDIA 3090 GPU with TensorRT 8.4.3, cuDNN 8.2.0, FP16, batch size=1, and without NMS.
