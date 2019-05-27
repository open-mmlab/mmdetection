# Lightweight SSD-lite based face detector

## Introduction

This config provides a very lightweight face detector based on the MobileNetV2
SSD-lite architecture. The detector uses only one SSD-lite head and
MobileNetV2 with width ratio 0.75x. As a result computational complexity of
the lightweight face detector is 0.51 GMAC, number of parameters is 1.03 M.


## Dataset

Face detector requires WIDER face dataset for training. You need to download
and extract it to the `data/WIDER` folder. Annotation in the VOC format
can be found in this [repo](https://github.com/sovrasov/wider-face-pascal-voc-annotations.git).
You should move the annotation files from `WIDER_train_annotations` `WIDER_val_annotations` folders
to the `Annotation` folders inside the corresponding directories `WIDER_train` and `WIDER_val`
The directory should be like this:

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── WIDER
│   │   ├── WIDER_train
│   |   │   ├── Annotations
│   │   ├── WIDER_val
│   |   │   ├── Annotations
```

## Results and Models

To download pre-trained MobileNetV2 backbone visit this [page](https://github.com/tonylins/pytorch-mobilenet-v2).

| Backbone  | Width factor  | Complexity (GMACS) | Parameters (M) | AP* (all faces) | AP* (faces > 30 pix)| AP* (faces > 60 pix) | AP* (faces > 100 pix)| Download |
|:---------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|
| MobileNetV2  | 0.75 |   0.51    | 1.03      | 0.305 | 0.768 | 0.867 | 0.927 | [model](https://drive.google.com/file/d/1jRi0hxxzIlfgEEoKHS_SkOD529y3kKNb/view?usp=sharing) |
| MobileNetV2  | 1.0  |   0.87    | 1.79      | 0.319 | 0.790 | 0.885 | 0.935 | [model](https://drive.google.com/file/d/1jjQBkLsNxfNR29UsJ2aeq1Ir4oZTuKEl/view?usp=sharing) |
*VOC AP on WIDER val with filtered faces by size threshold
