# Roboflow 100 Benchmark

> [Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark](https://arxiv.org/abs/2211.13523v3)

<!-- [Dataset] -->

## 摘要

目标检测模型的评估通常通过在一组固定的数据集上优化单一指标（例如 mAP），例如 Microsoft COCO 和 Pascal VOC。由于图像检索和注释成本高昂，这些数据集主要由在网络上找到的图像组成，并不能代表实际建模的许多现实领域，例如卫星、显微和游戏等，这使得很难确定模型学到的泛化程度。我们介绍了 Roboflow-100（RF100），它包括 100 个数据集、7 个图像领域、224,714 张图像和 805 个类别标签，超过 11,170 个标注小时。我们从超过 90,000 个公共数据集、6000 万个公共图像中提取了 RF100，这些数据集正在由计算机视觉从业者在网络应用程序 Roboflow Universe 上积极组装和标注。通过发布 RF100，我们旨在提供一个语义多样、多领域的数据集基准，帮助研究人员用真实数据测试模型的泛化能力。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/71b0eb6f-d710-4100-9fb1-9d5485e07fdb"/>
</div>

## 代码结构说明

```text
# 当前文件路径为 projects/RF100-Benchmark/
├── configs # 配置文件
│         ├── dino_r50_fpn_ms_8xb8_tweeter-profile.py
│         ├── faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py
│         └── tood_r50_fpn_ms_8xb8_tweeter-profile.py
├── README.md
├── README_zh-CN.md
├── rf100
└── scripts
    ├── create_new_config.py # 基于上述提供的配置生成其余 99 个数据集训练配置
    ├── datasets_links_640.txt # 数据集下载链接，来自官方 repo
    ├── download_dataset.py # 数据集下载代码，来自官方 repo
    ├── download_datasets.sh # 数据集下载脚本，来自官方 repo
    ├── labels_names.json # 数据集信息，来自官方 repo,不过由于有一些错误因此我们进行了修改
    ├── parse_dataset_link.py # 下载数据集需要，来自官方 repo
    ├── log_extract.py # 对训练的结果进行收集和整理
    └── dist_train.sh # 训练和评估启动脚本
    └── slurm_train.sh # slurm 训练和评估启动脚本
```

## 数据集准备

Roboflow 100 数据集是由 Roboflow 平台托管，并且在 [roboflow-100-benchmark](https://github.com/roboflow/roboflow-100-benchmark) 仓库中提供了详细的下载脚本。为了简单，我们直接使用官方提供的下载脚本。

在下载数据前，你首先需要在 Roboflow 平台注册账号，获取 API key。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/6126e69e-85ce-4dec-8e7b-936c4fae29a6"/>
</div>

```shell
export ROBOFLOW_API_KEY = 你的 Private API Key
```

同时你也应该安装 Roboflow 包。

```shell
pip install roboflow
```

最后使用如下命令下载数据集即可。

```shell
cd projects/RF100-Benchmark/
bash scripts/download_datasets.sh
```

下载完成后，会在当前目录下 `projects/RF100-Benchmark/` 生成 `rf100` 文件夹，其中包含了所有的数据集。其结构如下所示：

```text
# 当前文件路径为 projects/RF100-Benchmark/
├── README.md
├── README_zh-CN.md
└── scripts
    ├── datasets_links_640.txt
├── rf100
│    └── tweeter-profile
│    │    ├── train
|    |    |    ├── 0b3la49zec231_jpg.rf.8913f1b7db315c31d09b1d2f583fb521.jpg
|    |    |    ├──_annotations.coco.json
│    │    ├── valid
|    |    |    ├── 0fcjw3hbfdy41_jpg.rf.d61585a742f6e9d1a46645389b0073ff.jpg
|    |    |    ├──_annotations.coco.json
│    │    ├── test
|    |    |    ├── 0dh0to01eum41_jpg.rf.dcca24808bb396cdc07eda27a2cea2d4.jpg
|    |    |    ├──_annotations.coco.json
│    │    ├── README.dataset.txt
│    │    ├── README.roboflow.txt
│    └── 4-fold-defect
...
```

整个数据集一共需要 12.3G 存储空间。如果你不想一次性训练和评估所有模型，你可以修改 `scripts/datasets_links_640.txt` 文件，将你不想使用的数据集链接删掉即可。

Roboflow 100 数据集的特点如下图所示

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/e2693662-3d16-49a4-af0b-2a03be7e16b6"/>
</div>

如果想对数据集有个清晰的认识，可以查看 [roboflow-100-benchmark](https://github.com/roboflow/roboflow-100-benchmark) 仓库，其提供了诸多数据集分析脚本。

## 模型训练和评估

在准备好数据集后，可以一键开启单卡或者多卡训练。以 `faster-rcnn_r50_fpn` 算法为例

1. 单卡训练

```shell
# 当前位于 projects/RF100-Benchmark/
bash scripts/dist_train.sh configs/faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py 1
# 如果想指定保存路径
bash scripts/dist_train.sh configs/faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py 1 my_work_dirs
```

2. 分布式多卡训练

```shell
bash scripts/dist_train.sh configs/faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py 8
# 如果想指定保存路径
bash scripts/dist_train.sh configs/faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py 8 my_work_dirs
```

3. Slurm 训练

```shell
bash scripts/slurm_train.sh configs/faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py 8
# 如果想指定保存路径
bash scripts/slurm_train.sh configs/faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py 8 my_work_dirs
```

训练完成后会在当前路径下生成 `work_dirs` 文件夹，其中包含了训练好的模型权重和日志。

1. 为了方便用户调试或者只想训练特定的数据集，在 `scripts/*_train.sh` 中我们提供了 `DEBUG` 变量，你只需要设置为 1，并且在 `datasets_list` 变量中指定你想训练的数据集即可。
2. 考虑到由于各种原因，用户训练过程中可能出现某些数据集训练失败，因此我们提供了 `RETRY_PATH` 变量，你只需要传入 txt 数据集列表文件即可，程序会读取该文件中的数据集，然后只训练特定数据集。如果不提供则为全量数据集训练。

```shell
RETRY_PATH=failed_dataset_list.txt bash scripts/dist_train.sh configs/faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py 8 my_work_dirs
```

txt 文件中每一行代表一个数据集名称，示例如下(第 4 行的空行不可少)：

```text
acl-x-ray
tweeter-profile
abdomen-mri

```

上述 txt 文件你也可以采用后续介绍的 `log_extract.py` 脚本生成，而无需手动创建。

## 模型汇总

在模型训练好或者在训练中途你想对结果进行收集，你可以执行 `log_extract.py` 脚本，该脚本会将 `work_dirs` 下的信息收集并输出为 csv 和 xlsx 格式。

在运行脚本前，请确保安装了 `pandas` 和 `openpyxl`

```shell
python scripts/log_extract.py faster_rcnn --epoch 25 --work-dirs my_work_dirs
```

- 第一个输入参数是用于生成 csv 标题，因此你可以输入任意字符串，但是建议输入模型名称，方便后续查看。
- `--epoch` 参数是指模型训练 epoch 数，用于解析 log，默认我们是对每个数据集训练 100 epoch，但是配置中采用了 `RepeatDataset`，因此实际训练 epoch 是 25
- `--work-dirs` 是你训练模型保存的工作路径，默认是当前路径下的 `work_dirs` 文件夹

运行后会在 `my_work_dirs` 里面生成如下三个新文件

```text
时间戳_detail.xlsx # 100 个数据集的排序后详细信息
时间戳_sum.xlsx # 100 个数据集的汇总信息
时间戳_eval.csv # 100 个数据集的按照训练顺序评估结果
failed_dataset_list.txt # 失败数据集列表
```

目前我们提供了 `Faster RCNN、TOOD 和 DINO` 算法的评估结果(并没有进行精心的调参)。你也可以按照上述流程对自己的模型进行快速评估。

## 结果分析

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/86420903-36a8-410d-9251-4304b9704f7d"/>
</div>

💎 详情表，请直接访问 [结果](https://aicarrier.feishu.cn/drive/folder/QJ4rfqLzylIVTjdjYo3cunbinMh) 💎

为了确保对比公平且不存在特别的调参，`Faster RCNN、TOOD 和 DINO` 算法采用了相同的 epoch 和数据增强策略，并且都加载了 COCO 预训练权重，同时在训练中保存了验证集上性能最好的模型。其他说明如下所示：

- 为了加快训练速度，所有模型都是在 8 卡 GPU 上面训练。除了 DINO 算法在部分数据集上训练 OOM 外，其余所有模型和数据集都是在 8 张 3090 上训练
- 由于 'bacteria-ptywi', 'circuit-elements', 'marbles', 'printed-circuit-board', 'solar-panels-taxvb' 这 5 个数据集单张图片的 GT 框非常多导致 DINO 在 3090 上无法训练，因此这 5 个数据集我们在 A100 上进行训练

从上图来看，`DINO` 算法性能好于 `Faster RCNN 和 TOOD` 等传统 CNN 检测算法，说明 Transformer 算法在不同的领域或者不同数据量的情况下效果也是好于传统 CNN 类检测算法的，不过如果单独分析某些领域则不一定。

Roboflow 100 数据集本身也存在缺陷：

- 有些数据集训练图片数非常少，如果要统一超参进行 benchmark，可能会导致其性能很差
- 有些领域的部分数据集物体非常小且多，`Faster RCNN、TOOD 和 DINO` 在不进行特定调参情况下效果都非常差。针对这种情况，用户可以忽略这些数据集的结果
- 有些数据集标注的类别过于随意，如果想应用于图文检测类模型，则可能会存在性能低下的现象

最后需要说明：

1. 由于 100 个数据集比较多，我们无法对每个数据集进行检查，因此如果有不合理的地方，欢迎反馈，我们将尽快修复
2. 我们也提供了 `mAP_s` 等各种尺度的汇总结果，但是由于部分数据不存在这个尺度边界框，因为汇总时候我们忽略这些数据集

## 自定义算法进行 benchmark

如果用户想针对不同算法进行 Roboflow 100 Benchmark，你只需要在 `projects/RF100-Benchmark/configs` 文件夹新增算法配置即可。

注意：由于内部运行过程是通过将用户提供的配置中是以字符串替换的方式实现自定义数据集的功能，因此用户提供的配置必须是 `tweeter-profile` 数据集且必须包括 `data_root` 和 `class_name` 变量，否则程序会报错。

## 引用

```BibTeX
@misc{2211.13523,
Author = {Floriana Ciaglia and Francesco Saverio Zuppichini and Paul Guerrie and Mark McQuade and Jacob Solawetz},
Title = {Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark},
Year = {2022},
Eprint = {arXiv:2211.13523},
}
```
