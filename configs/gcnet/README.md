# GCNet for Object Detection

By [Yue Cao](http://yue-cao.me), [Jiarui Xu](http://jerryxu.net), [Stephen Lin](https://scholar.google.com/citations?user=c3PYmxUAAAAJ&hl=en), Fangyun Wei, [Han Hu](https://sites.google.com/site/hanhushomepage/).

We provide config files to reproduce the results in the paper for 
["GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond"](https://arxiv.org/abs/1904.11492) on COCO object detection.

## Introduction

**GCNet** is initially described in [arxiv](https://arxiv.org/abs/1904.11492). Via absorbing advantages of Non-Local Networks (NLNet) and Squeeze-Excitation Networks (SENet),  GCNet provides a simple, fast and effective approach for global context modeling, which generally outperforms both NLNet and SENet on major benchmarks for various recognition tasks.

## Citing GCNet

```
@article{cao2019GCNet,
  title={GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond},
  author={Cao, Yue and Xu, Jiarui and Lin, Stephen and Wei, Fangyun and Hu, Han},
  journal={arXiv preprint arXiv:1904.11492},
  year={2019}
}
```

## Results and models
The results on COCO 2017val are shown in the below table. 

| Backbone  |      Model       |    Context     | Lr schd | box AP | mask AP | Download  |
| :-------: | :--------------: | :------------: | :-----: | :----: | :-----: | :-------: |
| R-50-FPN  |       Mask       |       -        |   1x    |  37.2  |  33.8   | [model](https://1drv.ms/f/s!AkEXj14LxwVpcEkPyk8YWZVm9fo) |
|           |                  | GC(c3-c5, r16) |   1x    |  39.4  |  35.7   | [model](https://1drv.ms/f/s!AkEXj14LxwVpbSe8MYcjXToB090) |
|           |                  | GC(c3-c5, r4)  |   1x    |  39.9  |  36.2   | [model](https://1drv.ms/f/s!AkEXj14LxwVpaiEOLB67my2og3c) |
| R-101-FPN |       Mask       |       -        |   1x    |  39.8  |  36.0   | [model](https://1drv.ms/f/s!AkEXj14LxwVpcTx0JnBi1tuK4KU) |
|           |                  | GC(c3-c5, r16) |   1x    |  41.1  |  37.4   | [model](https://1drv.ms/f/s!AkEXj14LxwVpbqlp5GGBiZsoVaU) |
|           |                  | GC(c3-c5, r4)  |   1x    |  41.7  |  37.6   | [model](https://1drv.ms/f/s!AkEXj14LxwVpa-WTFi9kcyuWB8I) |
| X-101-FPN |       Mask       |       -        |   1x    |  41.2  |  37.3   | [model](https://1drv.ms/f/s!AkEXj14LxwVpcgwQlf8ok6IZahc) |
|           |                  | GC(c3-c5, r16) |   1x    |  42.4  |  38.0   | [model](https://1drv.ms/f/s!AkEXj14LxwVpb_4H8Q_bsmHZP7c) |
|           |                  | GC(c3-c5, r4)  |   1x    |  42.9  |  38.5   | [model](https://1drv.ms/f/s!AkEXj14LxwVpbLZR1Wu2JjMNyW0) |
| X-101-FPN |   Cascade Mask   |       -        |   1x    |  44.7  |  38.3   | [model](https://1drv.ms/f/s!Ak5O6Kz9hYOghnGJIdfM2fWlBsMR) |
|           |                  | GC(c3-c5, r16) |   1x    |  45.9  |  39.3   | [model](https://1drv.ms/f/s!Ak5O6Kz9hYOghm7gwd5yd4ZX-Dub) |
|           |                  | GC(c3-c5, r4)  |   1x    |  46.5  |  39.7   | [model](https://1drv.ms/f/s!Ak5O6Kz9hYOghk9roKJSNNpJGGsE) |
| X-101-FPN | DCN Cascade Mask |       -        |   1x    |  47.1  |  40.4   | [model](https://1drv.ms/f/s!Ak5O6Kz9hYOghmw3k3L6SbKl84Ex) |
|           |                  | GC(c3-c5, r16) |   1x    |  47.9  |  40.9   | [model](https://1drv.ms/f/s!Ak5O6Kz9hYOghm98OS64BAt0ZuOo) |
|           |                  | GC(c3-c5, r4)  |   1x    |  47.9  |  40.8   | [model](https://1drv.ms/f/s!Ak5O6Kz9hYOghlHmLvwSeF43ZISq) |

**Notes:**

- The `SyncBN` is added in the backbone for all models.
- `GC` denotes Global Context (GC) block is inserted after 1x1 conv of backbone. 
- `DCN` denotes replace 3x3 conv with 3x3 Deformable Convolution in `c3-c5` stages of backbone.
- `r4` and `r16` denote ratio 4 and ratio 16 in GC block respectively. 

