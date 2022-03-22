# SSD

> [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

<!-- [ALGORITHM] -->

## Abstract

We present a method for detecting objects in images using a single deep neural network. Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes. Our SSD model is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stage and encapsulates all computation in a single network. This makes SSD easy to train and straightforward to integrate into systems that require a detection component. Experimental results on the PASCAL VOC, MS COCO, and ILSVRC datasets confirm that SSD has comparable accuracy to methods that utilize an additional object proposal step and is much faster, while providing a unified framework for both training and inference. Compared to other single stage methods, SSD has much better accuracy, even with a smaller input image size. For 300×300 input, SSD achieves 72.1% mAP on VOC2007 test at 58 FPS on a Nvidia Titan X and for 500×500 input, SSD achieves 75.1% mAP, outperforming a comparable state of the art Faster R-CNN model.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143998553-4e12f681-6025-46b4-8410-9e2e1e53a8ec.png"/>
</div>

## Results and models of SSD

| Backbone | Size  | Style | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :------: | :---: | :---: | :-----: | :------: | :------------: | :----: | :------: |  :--------: |
|  VGG16   |  300  | caffe |  120e   |   9.9    |  43.7          |  25.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd/ssd300_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428.log.json) |
|  VGG16   |  512  | caffe |  120e   |   19.4   |  30.7          |  29.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd/ssd512_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210803_022849.log.json) |

## Results and models of SSD-Lite

|    Backbone    | Size  | Training from scratch | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :------------: | :---: | :-------------------: | :-----: | :------: | :------------: | :----: | :------: |  :--------: |
|  MobileNetV2   |  320  |        yes            |  600e   |   4.0    |     69.9       |  21.3  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627.log.json) |

## Notice

### Compatibility

In v2.14.0, [PR5291](https://github.com/open-mmlab/mmdetection/pull/5291) refactored SSD neck and head for more
flexible usage. If users want to use the SSD checkpoint trained in the older versions, we provide a scripts
`tools/model_converters/upgrade_ssd_version.py` to convert the model weights.

```bash
python tools/model_converters/upgrade_ssd_version.py ${OLD_MODEL_PATH} ${NEW_MODEL_PATH}

```

- OLD_MODEL_PATH: the path to load the old version SSD model.
- NEW_MODEL_PATH: the path to save the converted model weights.

### SSD-Lite training settings

There are some differences between our implementation of MobileNetV2 SSD-Lite and the one in [TensorFlow 1.x detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) .

1. Use 320x320 as input size instead of 300x300.
2. The anchor sizes are different.
3. The C4 feature map is taken from the last layer of stage 4 instead of the middle of the block.
4. The model in TensorFlow1.x is trained on coco 2014 and validated on coco minival2014, but we trained and validated the model on coco 2017. The mAP on val2017 is usually a little lower than minival2014 (refer to the results in TensorFlow Object Detection API, e.g., MobileNetV2 SSD gets 22 mAP on minival2014 but 20.2 mAP on val2017).

## Citation

```latex
@article{Liu_2016,
   title={SSD: Single Shot MultiBox Detector},
   journal={ECCV},
   author={Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
   year={2016},
}
```
