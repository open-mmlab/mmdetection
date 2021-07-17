# Rethinking Classification and Localization for Object Detection

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{wu2019rethinking,
    title={Rethinking Classification and Localization for Object Detection},
    author={Yue Wu and Yinpeng Chen and Lu Yuan and Zicheng Liu and Lijuan Wang and Hongzhi Li and Yun Fu},
    year={2019},
    eprint={1904.06493},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Results and models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-FPN     | pytorch |   1x    | 6.8      | 9.5            | 40.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/double_heads/dh_faster_rcnn_r50_fpn_1x_coco/dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/double_heads/dh_faster_rcnn_r50_fpn_1x_coco/dh_faster_rcnn_r50_fpn_1x_coco_20200130_220238.log.json) |
