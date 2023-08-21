# 学习训练和测试

## 训练

本节将介绍如何在支持的数据集上训练现有模型。
支持以下训练环境：

- CPU
- 单 GPU
- 单节点多 GPU
- 多节点

您还可以使用 Slurm 管理作业。

重要：

- 在训练过程中，您可以通过修改 `train_cfg` 来改变评估间隔。
  `train_cfg = dict(val_interval=10)`。这意味着每 10 个 epoch 对模型进行一次评估。
- 所有配置文件中的默认学习率为 8 个 GPU。
  根据[线性扩展规则](https://arxiv.org/abs/1706.02677)、
  如果在每个 GPU 上使用不同的 GPU 或图像，则需要设置与批次大小成比例的学习率、
  例如，8 个 GPU * 1 个图像/GPU 的学习率为 `lr=0.01`，16 个 GPU * 2 个图像/GPU 的学习率为 lr=0.04。
- 在训练过程中，日志文件和检查点将保存到工作目录、
  该目录由 CLI 参数 `--work-dir`指定。它默认使用 `./work_dirs/CONFIG_NAME`。
- 如果需要混合精度训练，只需指定 CLI 参数 `--amp`。

#### 1.在 CPU 上训练

该模型默认放在 cuda 设备上。
仅当没有 cuda 设备时，该模型才会放在 CPU 上。
因此，如果要在 CPU 上训练模型，则需要先 `export CUDA_VISIBLE_DEVICES=-1` 以禁用 GPU 可见性。
更多细节参见 [MMEngine](https://github.com/open-mmlab/mmengine/blob/ca282aee9e402104b644494ca491f73d93a9544f/mmengine/runner/runner.py#L849-L850).

```shell 脚本
CUDA_VISIBLE_DEVICES=-1 python tools/train.py ${CONFIG_FILE} [optional arguments]
```

在 CPU 上训练 MOT 模型 QDTrack 的示例：

```shell 脚本
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py
```

#### 2. 在单 GPU 上训练

如果您想在单 GPU 上训练模型, 您可以按照如下方法直接使用 `tools/train.py`.

```shell 脚本
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

您可以使用 `export CUDA_VISIBLE_DEVICES=$GPU_ID` 命令选择GPU.

在单 GPU 上训练 MOT 模型 QDTrack 的示例：

```shell 脚本
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py
```

#### 3. 在单节点多 GPU 上进行训练

我们提供了 `tools/dist_train.sh`，用于在多个 GPU 上启动训练。
基本用法如下。

```shell 脚本
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

如果您想在一台机器上启动多个作业、
例如，在拥有 8 个 GPU 的机器上启动 2 个 4-GPU 训练作业、
需要为每个作业指定不同的端口（默认为 29500），以避免通信冲突。

例如，可以在命令中设置端口如下。

```shell 脚本
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

在单节点多 GPU 上训练 MOT 模型 QDTrack 的示例：

```shell脚本
bash ./tools/dist_train.sh configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8
```

#### 4. 在多个节点上训练

如果使用以太网连接多台机器，只需运行以下命令即可：

在第一台机器上

```shell 脚本
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

在第二台机器上:

```shell script
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

如果没有 InfiniBand 等高速网络，速度通常会很慢。

#### 5. 使用 Slurm 进行训练

[Slurm](https://slurm.schedmd.com/)是一个用于计算集群的优秀作业调度系统。
在 Slurm 管理的集群上，您可以使用 `slurm_train.sh` 生成训练作业。
它支持单节点和多节点训练。

基本用法如下。

```shell 脚本
bash ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} ${GPUS}
```

使用 Slurm 训练 MOT 模型 QDTrack 的示例：

```shell脚本
PORT=29501 \
GPUS_PER_NODE=8 \
SRUN_ARGS="--quotatype=reserved" \
bash ./tools/slurm_train.sh \
mypartition \
mottrack
configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py
./work_dirs/QDTrack \
8
```

## 测试

本节将介绍如何在支持的数据集上测试现有模型。
支持以下测试环境：

- CPU
- 单 GPU
- 单节点多 GPU
- 多节点

您还可以使用 Slurm 管理作业。

重要：

- 在 MOT 中，某些算法（如 `DeepSORT`、`SORT`、`StrongSORT`）需要分别加载 `reid` 的权重和 `detector` 的权重。
  其他算法，如`ByteTrack`、`OCSORT`和`QDTrack`则不需要。因此，我们提供了 `--checkpoint`、`--detector` 和 `--reid`来加载权重。
- 我们提供了两种评估和测试模型的方法，即基于视频的测试和基于图像的测试。 有些算法如 `StrongSORT`, `Mask2former` 只支持基于视频的测试. 如果您的 GPU 内存无法容纳整个视频，您可以通过设置采样器类型来切换测试方式。
  例如
  基于视频的测试：`sampler=dict(type='DefaultSampler', shuffle=False, round_up=False)`
  基于图像的测试：`sampler=dict（type='TrackImgSampler'）`
- 您可以通过修改 evaluator 中的关键字 `outfile_prefix` 来设置结果保存路径。
  例如，`val_evaluator = dict(outfile_prefix='results/sort_mot17')`。
  否则，将创建一个临时文件，并在评估后删除。
- 如果您只想要格式化的结果而不需要评估，可以设置 `format_only=True`。
  例如，`test_evaluator = dict(type='MOTChallengeMetric', metric=['HOTA', 'CLEAR', 'Identity'], outfile_prefix='sort_mot17_results', format_only=True)`

#### 1. 在 CPU 上测试

模型默认在 cuda 设备上运行。
只有在没有 cuda 设备的情况下，模型才会在 CPU 上运行。
因此，如果要在 CPU 上测试模型，您需要 `export CUDA_VISIBLE_DEVICES=-1` 先禁用 GPU 可见性。

更多细节请参考[MMEngine](https://github.com/open-mmlab/mmengine/blob/ca282aee9e402104b644494ca491f73d93a9544f/mmengine/runner/runner.py#L849-L850).

```shell 脚本
CUDA_VISIBLE_DEVICES=-1 python tools/test_tracking.py ${CONFIG_FILE} [optional arguments]
```

在 CPU 上测试 MOT 模型 SORT 的示例：

```shell 脚本
CUDA_VISIBLE_DEVICES=-1 python tools/test_tracking.py configs/sort/sort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py --detector ${CHECKPOINT_FILE}
```

#### 2. 在单 GPU 上测试

如果您想在单 GPU 上测试模型，可以直接使用 `tools/test_tracking.py`，如下所示。

```shell 脚本
python tools/test_tracking.py ${CONFIG_FILE} [optional arguments]
```

您可以使用 `export CUDA_VISIBLE_DEVICES=$GPU_ID` 来选择 GPU。

在单 GPU 上测试 MOT 模型 QDTrack 的示例：

```shell 脚本
CUDA_VISIBLE_DEVICES=2 python tools/test_tracking.py configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py --detector ${CHECKPOINT_FILE}
```

#### 3. 在单节点多 GPU 上进行测试

我们提供了 `tools/dist_test_tracking.sh`，用于在多个 GPU 上启动测试。
基本用法如下。

```shell 脚本
bash ./tools/dist_test_tracking.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

在单节点多 GPU 上测试 MOT 模型 DeepSort 的示例：

```shell 脚本
bash ./tools/dist_test_tracking.sh configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8 --detector ${CHECKPOINT_FILE} --reid ${CHECKPOINT_FILE}
```

#### 4. 在多个节点上测试

您可以在多个节点上进行测试，这与 "在多个节点上进行训练 "类似。

#### 5. 使用 Slurm 进行测试

在 Slurm 管理的集群上，您可以使用 `slurm_test_tracking.sh` 生成测试作业。
它支持单节点和多节点测试。

基本用法如下。

```shell 脚本
[GPUS=${GPUS}] bash tools/slurm_test_tracking.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} [optional arguments]
```

使用 Slurm 测试 VIS 模型 Mask2former 的示例：

```shell 脚本
GPUS=8
bash tools/slurm_test_tracking.sh \
mypartition \
vis \
configs/mask2former_vis/mask2former_r50_8xb2-8e_youtubevis2021.py \
--checkpoint ${CHECKPOINT_FILE}
```
