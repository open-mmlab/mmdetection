# SOLO: Segmenting Objects by Locations

## Introduction

```
@inproceedings{wang2020solo,
  title     =  {{SOLO}: Segmenting Objects by Locations},
  author    =  {Wang, Xinlong and Kong, Tao and Shen, Chunhua and Jiang, Yuning and Li, Lei},
  booktitle =  {Proc. Eur. Conf. Computer Vision (ECCV)},
  year      =  {2020}
}
```

## Results and Models

### SOLO

| Backbone  | Style   | MS train | Lr schd | Mem (GB) | Inf time (fps) | mask AP | Download |
|:---------:|:-------:|:--------:|:-------:|:--------:|:--------------:|:------:|:--------:|
| R-50      | pytorch   | N        | 1x      |       |            |  33.1  |  [model]() &#124; [log]() |
| R-50      | pytorch   | Y        | 3x      |       |          |    |  [model]() &#124; [log]() |

### Decoupled SOLO

| Backbone  | Style   | MS train | Lr schd | Mem (GB) | Inf time (fps) | mask AP | Download |
|:---------:|:-------:|:--------:|:-------:|:--------:|:--------------:|:------:|:--------:|
| R-50      | pytorch   | N        | 1x      |       |            |  34.0  |  [model]() &#124; [log]() |
| R-50      | pytorch   | Y        | 3x      |       |          |    |  [model]() &#124; [log]() |
