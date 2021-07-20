# YOLOX
## Introduction

YOLOX is an anchor-free version of YOLO, with a simpler design but better performance! It aims to bridge the gap between research and industrial communities.
For more details, please refer to [Arxiv report](https://arxiv.org/abs/2107.08430).

## Results and Models

### Standard Models

|Model |size |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/yolox_s.py)    |640  |39.6      |9.8     |9.0 | 26.8 | [Download](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EW62gmO2vnNNs5npxjzunVwB9p307qqygaCkXdTO88BLUg?e=NMTQYw) |
|[YOLOX-m](./exps/yolox_m.py)    |640  |46.4      |12.3     |25.3 |73.8| [Download](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ERMTP7VFqrVBrXKMU7Vl4TcBQs0SUeCT7kvc-JdIbej4tQ?e=1MDo9y) |
|[YOLOX-l](./exps/yolox_l.py)    |640  |50.0  |14.5 |54.2| 155.6 | [Download](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EWA8w_IEOzBKvuueBqfaZh0BeoG5sVzR-XYbOJO4YlOkRw?e=wHWOBE) |
|[YOLOX-x](./exps/yolox_x.py)   |640  |**51.2**      | 17.3 |99.1 |281.9 | [Download](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdgVPHBziOVBtGAXHfeHI5kBza0q9yyueMGdT0wXZfI1rQ?e=tABO5u) |
|[YOLOX-Darknet53](./exps/yolov3.py)   |640  | 47.4      | 11.1 |63.7 | 185.3 | [Download](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EZ-MV1r_fMFPkPrNjvbJEMoBLOLAnXH-XKEB77w8LhXL6Q?e=mf6wOc) |

### Light Models

|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/nano.py) |416  |25.3  | 0.91 |1.08 | [Download](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdcREey-krhLtdtSnxolxiUBjWMy6EFdiaO9bdOwZ5ygCQ?e=yQpdds) |
|[YOLOX-Tiny](./exps/yolox_tiny.py) |416  |31.7 | 5.06 |6.45 | [Download](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EYtjNFPqvZBBrQ-VowLcSr4B6Z5TdTflUsr_gO2CwhC3bQ?e=SBTwXj) |

## Citation

```latex
@article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
