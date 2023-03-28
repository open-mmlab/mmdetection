# 背景

标注数据是一项耗费精力和财力的任务。本文介绍了如何使用MMDetection中的RTMDet算法联合Label-Studio软件进行标注。具体来说，使用RTMDet预测图片生成标注，然后使用Label-Studio进行微调，社区用户可以参考此流程和方法，将其应用到其他领域。

- RTMDet：RTMDet是OpenMMLab自研的高精度单阶段的目标检测算法，开源于MMDetection目标检测工具箱中，其开源协议为 Apache 2.0，工业界的用户可以不受限的免费使用。
- Label Studio 是一款优秀的标注软件，覆盖图像分类、目标检测、分割等领域数据集标注的功能。

本文使用将[喵喵](https://download.openmmlab.com/mmyolo/data/cat_dataset.zip)的图片，进行半自动化标注。

# 环境配置

创建虚拟环境：

```shell
conda create -n rtmdet python=3.9 -y
conda activate rtmdet
```

安装PyTorch

```shell
# Linux and Windows CPU only
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
# Linux and Windows CUDA 11.3
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# OSX
pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1
```

安装mmc

```shell
pip install -U openmim
mim install "mmcv>=2.0.0rc0"
# 安装mmcv的过程中会自动安装mmengine
```

为了避免软件版本升级而导致本文档后续不可用，所以在本文中将指定版本。

安装mmdetection

```shell
git clone https://github.com/open-mmlab/mmdetection -b dev-3.x
cd mmdetection
pip install -v -e .
```

安装Label-Studio和label-studio-ml-backend

```shell
# 安装 label-studio 需要一段时间
pip install label-studio==1.7.2
pip install label-studio-ml==1.0.9
```

下载rtmdet权重

```shell
cd path/to/mmetection
mkdir work_dirs
cd work_dirs
wget https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth
```

启动RTMDet后端推理服务

```shell
cd path/to/mmetection
label-studio-ml start projects/Label-Studio/backend-template --with \
config_file=configs/rtmdet/rtmdet_m_8xb32-300e_coco.py \
checkpoint_file=./work_dirs/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth \
device=cpu \
--port 8003
# device=cpu 为使用CPU推理，如果使用GPU推理，将cpu替换为cuda:0
```

启动Label-Studio 网页服务

```shell
label-studio start
```

```shell
airplane
apple
backpack
banana
baseball_bat
baseball_glove
bear
bed
bench
bicycle
bird
boat
book
bottle
bowl
broccoli
bus
cake
car
carrot
cat
cell_phone
chair
clock
couch
cow
cup
dining_table
dog
donut
elephant
fire_hydrant
fork
frisbee
giraffe
hair_drier
handbag
horse
hot_dog
keyboard
kite
knife
laptop
microwave
motorcycle
mouse
orange
oven
parking_meter
person
pizza
potted_plant
refrigerator
remote
sandwich
scissors
sheep
sink
skateboard
skis
snowboard
spoon
sports_ball
stop_sign
suitcase
surfboard
teddy_bear
tennis_racket
tie
toaster
toilet
toothbrush
traffic_light
train
truck
tv
umbrella
vase
wine_glass
zebra
```
