# 学习更多与配置相关的事

我们用 python 文档作为我们的配置系统。你可以在 `MMDetection/configs` 底下找到所有已提供的配置文件。

我们把模块化和继承化设计融入我们的配置系统，这使我们很方便去进行各种实验。如果你想查看相关的配置文件，你可以跑 `python tools/misc/print_config.py /PATH/TO/CONFIG` 去看完整的详细配置。

## 完整配置的简要说明

一个完整的配置通常包含以下主要的字段：

`model`：一个模型的基本配置，包含 `data_preprocessor`、`detector`、`motion` 之类的模块，还有 `train_cfg`、`test_cfg` 等等；

`train_dataloader`：训练数据集的配置，通常包含 `batch_size`、 `num_workers`、 `sampler`、 `dataset` 等等；

`val_dataloader`：验证数据集的配置，与训练数据集的配置类似；

`test_dataloader`：测试数据集的配置，与训练数据集的配置类似；

`val_evaluator`：验证评估器的配置，例如 `type='MOTChallengeMetrics'` 是 MOT 任务里面的测量标准；

`test_evaluator`：测试评估器的配置，与验证评估器的配置类似；

`train_cfg`：训练循环的配置，例如 `type='EpochBasedTrainLoop'` ；

`val_cfg`：验证循环的配置，例如 `type='VideoValLoop'` ；

`test_cfg`：测试循环的配置，例如 `type='VideoTestLoop'` ；

`default_hooks`：默认鱼钩的配置，包含计时器、日志、参数调度程序、检查点、样本种子、可视化；

`vis_backends`：可视化后端的配置，默认使用 `type='LocalVisBackend'` ；

`visualizer`：可视化工具的配置，例如MOT任务使用 `type='TrackLocalVisualizer'` ；

`param_scheduler`：参数调度程序的配置，通常里面设置学习率调度程序；

`optim_wrapper`：优化器封装的配置，包含优化相关的信息，例如优化器、梯度剪裁等；

`load_from`：加载预训练模型的路径；

`resume`：布尔值，如果是 `True` ，会从 `load_from` 加载模型的检查点，训练会恢复至检查点的迭代次数。

## 通过脚本参数修改配置

当使用 `tools/train.py` 或 `tools/test_trackin.py` 执行任务时，可以指定 `--cfg-options` 来就地修改配置。我们举几个例子如下。有关更多详细信息，请参阅[MMEngine](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html)。

### 更新 dict 链的配置键

可以按照原始配置中 `dict` 键的顺序指定配置选项，例如，设置 `--cfg-options model.detector.backbone.norm_eval=False` 会将模型主干中的所有 `BN` 模块更改为训练模式。

### 更新配置列表中的关键字

一些配置的 `dict` 关键字会以列表的形式组成，例如，测试管道中的 `test_dataloader.dataset.pipeline` 以列表形式出现，即 `[dict(type='LoadImageFromFile'), ...]`。如果你想在测试管道中将 `LoadImageFromFile` 更改为 `LoadImageFromWebcam`，可以设置 `--cfg-options test_dataloader.dataset.pipeline.0.type=LoadImageFromWebcam`。

### 更新列表/元组的值

要被更新的可能是一个列表或一个元组，例如，你可以通过指定 `--cfg options model.data_processor.mean=[0,0,0]` 来更改 `data_preprocessor` 的平均值的关键字。请注意，指定值内不允许有空格。

## 配置文件结构

`config/_base_` 下有三种基本组件类型，即数据集、模型和默认运行时间。可以用它们来轻松构建许多方法，例如 `SORT`，`DeepSORT`。由 `_base_` 中的组件组成的配置称为基元。

对于同一文件夹下的配置文件，建议只有一个基元配置文件。其他配置文件都应该从基元配置文件继承基本结构，这样，继承级别的最大值为 3。

为了便于理解，我们建议贡献者继承现有的方法。例如，如果在 `Faster R-CNN` 的基础上进行了一些修改，用户可以首先通过指定 `_base_ = ../_base_/models/faster-rcnn_r50-dc5.py` 来继承基本的 `Faster R-CNN` 结构，然后修改配置文件中的必要字段。

如果你正在构建一个与任何现有方法都不共享结构的全新方法，则可以在 `configs` 下创建一个新文件夹 method_name。

有关详细文档，请参阅[MMEngine](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html)。

## 配置命名风格

我们根据以下风格去命名配置文件，建议贡献者遵从相同风格。

`{method}_{module}_{train_cfg}_{train_data}_{test_data}`

`{method}`: 方法名称，例如 `sort`；

`{module}`: 方法的基本模块，例如 `faster-rcnn_r50_fpn`；

`{train_cfg}`: 训练配置通常包含批量大小、迭代次数等，例如 `8xb4-80e`；

`{train_data}`: 训练数据集，例如 `mot17halftrain`；

`{test_data}`: 测试数据集，例如 `test-mot17halfval`。

## 常问问题

### 忽略基本配置中的某些字段

有时候你可以设置 `_delete_=True` 去忽略基本配置中的一些字段，你可以参考[MMEngine](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html)进行简单说明。

### 跟踪数据结构介绍

#### 优点和新功能

在 `mmdetection` 跟踪任务中，我们使用视频来组织数据集，并使用 `TrackDataSample` 来描述数据集信息。

基于视频组织，我们提供了 `transform UniformRefFrameSample` 来对关键帧和参考帧进行采样，并使用 `TransformBroadcaster` 进行剪辑训练。

在某种程度上，`TrackDataSample` 可以被视为多个 `DetDataSample` 的包装器。它包含一个 `video_data_samples`，这是一个以 `DetDataSample` 组成的列表，里面每个 `DetDataSample` 对应一个帧。此外，它的元信息包括关键帧的索引和参考帧的索引，用与剪辑训练。

得益于基于视频的数据组织，整个视频可以直接被测试。这种方式更简洁直观。如果你的 GPU 内存无法容纳整个视频，我们还提供基于图像的测试方法。

## 要做的事

`StrongSORT`、`Mask2Former` 等算法不支持基于视频的测试，这些算法对 GPU 内存提出了挑战，我们将来会优化这个问题。

现在，我们不支持像 `MOT Challenge dataset` 这样的基于视频的数据集和像 `Crowdhuman` 用于 `QDTrack` 算法这样的基于图像的数据集进行联合训练。我们将来会优化这个问题。
