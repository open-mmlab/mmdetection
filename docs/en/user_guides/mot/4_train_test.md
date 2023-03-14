# Learn to train and test

## Train

This section will show how to train existing models on supported datasets.
The following training environments are supported:

- CPU
- single GPU
- single node multiple GPUs
- multiple nodes

You can also manage jobs with Slurm.

Important:

- You can change the evaluation interval during training by modifying the `train_cfg` as
  `train_cfg = dict(val_interval=10)`. That means evaluating the model every 10 epochs.
- The default learning rate in all config files is for 8 GPUs.
  According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677),
  you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU,
  e.g., `lr=0.01` for 8 GPUs * 1 img/gpu and lr=0.04 for 16 GPUs * 2 imgs/gpu.
- During training, log files and checkpoints will be saved to the working directory,
  which is specified by CLI argument `--work-dir`. It uses `./work_dirs/CONFIG_NAME` as default.
- If you want the mixed precision training, simply specify CLI argument `--amp`.

#### 1. Train on CPU

The model is default put on cuda device.
Only if there are no cuda devices, the model will be put on cpu.
So if you want to train the model on CPU, you need to `export CUDA_VISIBLE_DEVICES=-1` to disable GPU visibility first.
More details in [MMEngine](https://github.com/open-mmlab/mmengine/blob/ca282aee9e402104b644494ca491f73d93a9544f/mmengine/runner/runner.py#L849-L850).

```shell script
CUDA_VISIBLE_DEVICES=-1 python tools/train.py ${CONFIG_FILE} [optional arguments]
```

An example of training the MOT model QDTrack on CPU:

```shell script
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py
```

#### 2. Train on single GPU

If you want to train the model on single GPU, you can directly use the `tools/train.py` as follows.

```shell script
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

You can use `export CUDA_VISIBLE_DEVICES=$GPU_ID` to select the GPU.

An example of training the MOT model QDTrack on single GPU:

```shell script
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py
```

#### 3. Train on single node multiple GPUs

We provide `tools/dist_train.sh` to launch training on multiple GPUs.
The basic usage is as follows.

```shell script
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

If you would like to launch multiple jobs on a single machine,
e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

For example, you can set the port in commands as follows.

```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

An example of training the MOT model QDTrack on single node multiple GPUs:

```shell script
bash ./tools/dist_train.sh configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8
```

#### 4. Train on multiple nodes

If you launch with multiple machines simply connected with ethernet, you can simply run following commands:

On the first machine:

```shell script
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

On the second machine:

```shell script
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

Usually it is slow if you do not have high speed networking like InfiniBand.

#### 5. Train with Slurm

[Slurm](https://slurm.schedmd.com/) is a good job scheduling system for computing clusters.
On a cluster managed by Slurm, you can use `slurm_train.sh` to spawn training jobs.
It supports both single-node and multi-node training.

The basic usage is as follows.

```shell script
bash ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} ${GPUS}
```

An example of training the MOT model QDTrack with Slurm:

```shell script
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

## Test

This section will show how to test existing models on supported datasets.
The following testing environments are supported:

- CPU
- single GPU
- single node multiple GPUs
- multiple nodes

You can also manage jobs with Slurm.

Important:

- You can set the results saving path by modifying the key `outfile_prefix` in evaluator.
  For example, `val_evaluator = dict(outfile_prefix='results/sort_mot17')`.
  Otherwise, a temporal file will be created and will be removed after evaluation.
- If you just want the formatted results without evaluation, you can set `format_only=True`.
  For example, `test_evaluator = dict(type='MOTChallengeMetric', metric=['HOTA', 'CLEAR', 'Identity'], outfile_prefix='sort_mot17_results', format_only=True)`

#### 1. Test on CPU

The model is default put on cuda device.
Only if there are no cuda devices, the model will be put on cpu.
So if you want to test the model on CPU, you need to `export CUDA_VISIBLE_DEVICES=-1` to disable GPU visibility first.
More details in [MMEngine](https://github.com/open-mmlab/mmengine/blob/ca282aee9e402104b644494ca491f73d93a9544f/mmengine/runner/runner.py#L849-L850).

```shell script
CUDA_VISIBLE_DEVICES=-1 python tools/test_tracking.py ${CONFIG_FILE} [optional arguments]
```

An example of testing the MOT model SORT on CPU:

```shell script
CUDA_VISIBLE_DEVICES=-1 python tools/test_tracking.py configs/sort/sort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py
```

#### 2. Test on single GPU

If you want to test the model on single GPU, you can directly use the `tools/test_tracking.py` as follows.

```shell script
python tools/test_tracking.py ${CONFIG_FILE} [optional arguments]
```

You can use `export CUDA_VISIBLE_DEVICES=$GPU_ID` to select the GPU.

An example of testing the MOT model QDTrack on single GPU:

```shell script
CUDA_VISIBLE_DEVICES=2 python tools/test_tracking.py configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py --checkpoint https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635-76f295ef.pth
```

#### 3. Test on single node multiple GPUs

We provide `tools/dist_test_tracking.sh` to launch testing on multiple GPUs.
The basic usage is as follows.

```shell script
bash ./tools/dist_test_tracking.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

An example of testing the MOT model DeepSort on single node multiple GPUs:

```shell script
bash ./tools/dist_test_tracking.sh configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8 --detector https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth --reid https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth
```

#### 4. Test on multiple nodes

You can test on multiple nodes, which is similar with "Train on multiple nodes".

#### 5. Test with Slurm

On a cluster managed by Slurm, you can use `slurm_test_tracking.sh` to spawn testing jobs.
It supports both single-node and multi-node testing.

The basic usage is as follows.

```shell script
bash ./tools/slurm_test_tracking.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${GPUS}
```

An example of testing the MOT model QDTrack with Slurm:

```shell script
PORT=29501 \
GPUS_PER_NODE=8 \
SRUN_ARGS="--quotatype=reserved" \
bash ./tools/slurm_test_tracking.sh \
mypartition \
mottrack \
configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py \
8 \
--checkpoint https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635-76f295ef.pth
```
