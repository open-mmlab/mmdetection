# SSD: Single Shot MultiBox Detector

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{Liu_2016,
   title={SSD: Single Shot MultiBox Detector},
   journal={ECCV},
   author={Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
   year={2016},
}
```

## Results and models

| Backbone | Size  | Style | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :------: | :---: | :---: | :-----: | :------: | :------------: | :----: | :------: |  :--------: |
|  VGG16   |  300  | caffe |  120e   |   9.9    |  43.7          |  25.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd/ssd300_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210604_193052-b61137df.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210604_193052.log.json) |
|  VGG16   |  512  | caffe |  120e   |   19.4   |  30.7          |  29.8  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd/ssd512_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210604_111835-d3eba047.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210604_111835.log.json) |

## Notice

In v2.14.0, [PR5291](https://github.com/open-mmlab/mmdetection/pull/5291) refactored SSD neck and head for more
flexible usage. If users want to use the SSD checkpoint trained in the older versions, we provide a scripts
`tools/model_converters/upgrade_ssd_version.py` to convert the model weights.

```bash
python tools/model_converters/upgrade_ssd_version.py ${OLD_MODEL_PATH} ${NEW_MODEL_PATH}

```

- OLD_MODEL_PATH: the path to load the old version SSD model.
- NEW_MODEL_PATH: the path to save the converted model weights.
