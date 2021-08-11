



## CenterNet2 - MMDetection
This repository is our reproduced implemention of [Probabilistic two-stage detection](https://arxiv.org/pdf/2103.07461.pdf), and submitted for [OpenMMLab Algorithm Ecological Challenge](https://openmmlab.com/competitions/algorithm-2021). You can also refer to our clean version of [repo](https://github.com/smart-car-lab/Centernet2-mmdetction/) or the [PR](https://github.com/open-mmlab/mmdetection/pull/5854
) to the official mmdetection for details. 


### Result and model details

 **1. basic settings**
 - backbone：ResNet-50
 - neck：FPN
 - rpn_head: **CustomCenterNetHead**
 - roi_head:  **CustomCascadeRoIHead**

**2. parameters and resluts:**

We change ResNet-FPN to Retinaanet Style, some configurations differ from the original version of MMDetection are as follows：
 - ResNet-50：
 out_indices=(1, 2, 3)
 - FPN： 
 in_channels=[256, 512, 1024, 2048]
 add_extra_convs='on_output'
 relu_before_extra_convs=True
 
please refer the config file [centernet2_cascade_r50_fpn.py](https://github.com/smart-car-lab/Centernet2-mmdetction/blob/main/configs/centernet2_cascade_r50_fpn.py) for more details.

**Result** by this implementation:
 
| name | bbox_mAP|bbox_mAP_50|bbox_mAP_75|bbox_mAP_l|bbox_mAP_m|bbox_mAP_s|
|--|--|--|--|--|--|--|
| CenterNet2 | 40.5 | 56.8 | 44.6 | 21.2 | 44.1 | 55.6 |

log and model:
| name | backbone | schedule | mAP | Log | Model |
|--|--|--|--|--|--|
| CenterNet2 | R50-FPN | 1x | 40.5 | [log](http) | [CenterNet2_1x](https://pan.baidu.com/s/1yUnpu146aDk558vmhiNfxg)[9doa]|
### How To Use？
What is MMCV？
MMCV is a foundational library for computer vision research and supports many research projects as below:

 - [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
 - [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
 - [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab’s next-generation platform for general 3D object detection.
 - [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
 - [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab’s next-generation action understanding toolbox and benchmark.
 - [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.


---

Our project is based on [MMDetection](https://github.com/open-mmlab/mmdetection), please refer [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) for the env installation and basic usage of MMDetection. The env details as follows:

- Ubuntu 18.04
- Python: 3.7 
- PyTorch: 1.7.1 + TorchVision: 0.8.2
- NVCC: Build cuda_11.0_bu.TC445_37.28845127_0
- GCC: 7.5.0
- OpenCV: 4.5.2
- MMCV: 1.3.8
- MMDetection: 2.13.0+81310d6




1. **clone our repo to your workstation**

```
git clone https://github.com/Jacky-gsq/Centernet2-mmdet
```

2. **copy follwing files to the directory of mmdetection project**

```bash
cd CenterNet2-MMDetection
mv ./configs/centernet2 ${your path to mmdetection}/configs/
mv ./configs/_base_/models/centernet2_cascade_r50_fpn.py ${your path to mmdetection}/configs/_base_/models/
mv ./mmdet/models/detectors/centernet2.py ${your path to mmdetection}/mmdet/models/detectors/
mv ./mmdet/models/dense_heads/custom_centernet_head.py ${your path to mmdetection}/mmdet/models/dense_heads/
mv ./mmdet/models/roi_heads/custom_cascade_roi_head.py ${your path to mmdetection}/mmdet/models/roi_heads/
mv ./mmdet/models/losses/gaussian_focal_loss.py ${your path to mmdetection}/mmdet/models/losses/
```


3. **register and import module in  `__init__.py`**

*mmdetection/models/detectors/\_\_init\_\_.py*


```python
...
from .centernet2 import CenterNet2

__all__ = [
    ..., 'CenterNet2'
]
```


*mmdetection/models/dense_heads/\_\_init\_\_.py*

```python
...
from .custom_centernet_head import CustomCenterNetHead

__all__ = [
    ..., 'CustomCenterNetHead'
]
```
*mmdetection/models/roi_heads/\_\_init\_\_.py*

```python
...
from .custom_cascade_roi_head import CustomCascadeRoIHead

__all__ = [
    ..., 'CustomCascadeRoIHead'
]
```

*mmdetection/models/roi_heads/\_\_init\_\_.py*

```python
...

__all__ = [
    ..., 'CustomGaussianFocalLoss'
]
```

## Train and Test
- **prepare coco dataset**

```bash
cd ${your path to mmdetection}
mkdir data && cd data
ln -s ${your path to coco dataset} ./
```

- **train**

```bash
# single-gpu
cd ${your path to mmdetection}/tools
python ./train.py  ../configs/centernet2/centernet2_cascade_res50_fpn_1x_coco.py [optional arguments]
# multi-gpu
./dist_train.sh ../configs/centernet2/centernet2_cascade_res50_fpn_1x_coco.py ${GPU_NUM} [optional arguments]
```

- **test**
```bash
# single-gpu
cd ${your path to mmdetection}/tools
python ./test.py  ../configs/centernet2/centernet2_cascade_res50_fpn_1x_coco.py ${CHECKPOINT_FILE} [optional arguments]
# multi-gpu
./dist_test.sh ../configs/centernet2/centernet2_cascade_res50_fpn_1x_coco.py ${CHECKPOINT_FILE} ${GPU_NUM} --out ${RESULT_FILE} [optional arguments]
```








## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}

@inproceedings{zhou2021probablistic,
  title={Probabilistic two-stage detection},
  author={Zhou, Xingyi and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle={arXiv preprint arXiv:2103.07461},
  year={2021}
}
```
