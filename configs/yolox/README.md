<div align="center"><img src="assets/logo.png" width="350"></div>
<img src="assets/demo.png" >

## Introduction
YOLOX is an anchor-free version of YOLO, with a simpler design but better performance! It aims to bridge the gap between research and industrial communities.
For more details, please refer to our [Arxiv report]().

<img src="assets/git_fig.png" width="1000" >

## Updates!!
* 【2020/07/19】 We have released our technical report on [Arxiv]().

## Comming soon
- [ ] YOLOX-P6 and larger model.
- [ ] Obj365 pretrain.
- [ ] Transformer modules.
- [ ] More features in need.

## Benchmark

#### Standard Models.
|Model |size |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(B)| weights |
| ------        |:---: | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/yolox_s.py)    |640  |39.6      |9.8     |9.0 | 26.8 | [Download](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EW62gmO2vnNNs5npxjzunVwB9p307qqygaCkXdTO88BLUg?e=NMTQYw) |
|[YOLOX-m](./exps/yolox_m.py)    |640  |46.4      |12.3     |25.3 |73.8| [Download](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ERMTP7VFqrVBrXKMU7Vl4TcBQs0SUeCT7kvc-JdIbej4tQ?e=1MDo9y) |
|[YOLOX-l](./exps/yolox_l.py)    |640  |50.0  |14.5 |54.2| 155.6 | [Download](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EWA8w_IEOzBKvuueBqfaZh0BeoG5sVzR-XYbOJO4YlOkRw?e=wHWOBE) |
|[YOLOX-x](./exps/yolox_x.py)   |640  |**51.2**      | 17.3 |99.1 |281.9 | [Download](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdgVPHBziOVBtGAXHfeHI5kBza0q9yyueMGdT0wXZfI1rQ?e=tABO5u) |
|[YOLOX-Darknet53](./exps/yolov3.py)   |640  | 47.4      | 11.1 |63.7 | 185.3 | [Download](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EZ-MV1r_fMFPkPrNjvbJEMoBLOLAnXH-XKEB77w8LhXL6Q?e=mf6wOc) |

#### Light Models.
|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(B)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/nano.py) |416  |25.3  | 0.91 |1.08 | [Download](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdcREey-krhLtdtSnxolxiUBjWMy6EFdiaO9bdOwZ5ygCQ?e=yQpdds) |
|[YOLOX-Tiny](./exps/yolox_tiny.py) |416  |31.7 | 5.06 |6.45 | [Download](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EYtjNFPqvZBBrQ-VowLcSr4B6Z5TdTflUsr_gO2CwhC3bQ?e=SBTwXj) |

## Quick Start

<details>
<summary>Installation</summary>

Step1. Install YOLOX.
```shell
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd yolox
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .  # or  python3 setup.py develop
```
Step2. Install [apex](https://github.com/NVIDIA/apex).

```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
Step3. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

</details>

</details>

<details>
<summary>Demo</summary>

Step1. Download a pretrained model from the benchmark table.

Step2. Use either -n or -f to specify your detector's config. For example:

```shell
python tools/demo.py image -n yolox-s -c /path/to/your/yolox_s.pth.tar --path assets/dog.jpg --conf 0.3 --nms 0.65 --tsize 640 --save_result
```
or
```shell
python tools/demo.py image -f exps/yolox_s.py -c /path/to/your/yolox_s.pth.tar --path assets/dog.jpg --conf 0.3 --nms 0.65 --tsize 640 --save_result
```
Demo for video:
```shell
python tools/demo.py video -n yolox-s -c /path/to/your/yolox_s.pth.tar --path /path/to/your/video --conf 0.3 --nms 0.65 --tsize 640 --save_result
```


</details>

<details>
<summary>Reproduce our results on COCO</summary>

Step1. Prepare dataset
```shell
cd <YOLOX_HOME>
mkdir datasets
ln -s /path/to/your/COCO ./datasets/COCO
```

Step2. Reproduce our results on COCO by specifying -n:

```shell
python tools/train.py -n yolox-s -d 8 -b 64 --fp16 -o
                         yolox-m
                         yolox-l
                         yolox-x
```
* -d: number of gpu devices
* -b: total batch size, the recommended number for -b is num_gpu * 8
* --fp16: mixed precision training

When using -f, the above commands are equivalent to:

```shell
python tools/train.py -f exps/base/yolox-s.py -d 8 -b 64 --fp16 -o
                         exps/base/yolox-m.py
                         exps/base/yolox-l.py
                         exps/base/yolox-x.py
```

</details>


<details>
<summary>Evaluation</summary>

We support batch testing for fast evaluation:

```shell
python tools/eval.py -n  yolox-s -c yolox_s.pth.tar -b 64 -d 8 --conf 0.001 [--fp16] [--fuse]
                         yolox-m
                         yolox-l
                         yolox-x
```
* --fuse: fuse conv and bn
* -d: number of GPUs used for evaluation. DEFAULT: All GPUs available will be used.
* -b: total batch size across on all GPUs

To reproduce speed test, we use the following command:
```shell
python tools/eval.py -n  yolox-s -c yolox_s.pth.tar -b 1 -d 1 --conf 0.001 --fp16 --fuse
                         yolox-m
                         yolox-l
                         yolox-x
```

</details>


<details open>
<summary>Toturials</summary>

*  [Training on custom data](docs/train_custom_data.md).

</details>

## Deployment


1.  [ONNX: Including ONNX export and an ONNXRuntime demo.](./demo/ONNXRuntime)
2.  [TensorRT in both C++ and Python](./demo/TensorRT)
3.  [NCNN in C++](./demo/ncnn/android)
4.  [OpenVINO in both C++ and Python](./demo/OpenVINO)

## Citing YOLOX
If you use YOLOX in your research, please cite our work by using the following BibTeX entry:
  
```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2021:xxxx},
  year={2021}
}
```
