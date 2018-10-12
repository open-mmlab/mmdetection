# Benchmark and Model Zoo

## Environment

### Hardware

- 8 NVIDIA Tesla V100 GPUs
- Intel Xeon 4114 CPU @ 2.20GHz

### Software environment

- Python 3.6 / 3.7
- PyTorch 0.4.1
- CUDA 9.0.176
- CUDNN 7.0.4
- NCCL 2.1.15


## Common settings

- All baselines were trained using 8 GPU with a batch size of 16 (2 images per GPU).
- All models were trained on `coco_2017_train`, and tested on the `coco_2017_val`.
- We use distributed training and BN layer stats are fixed.
- We adopt the same training schedules as Detectron. 1x indicates 12 epochs and 2x indicates 24 epochs, which corresponds to slightly less iterations than Detectron and the difference can be ignored.
- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo.
- We report the training GPU memory as the maximum value of `torch.cuda.max_memory_cached()`
for all 8 GPUs. Note that this value is usually less than what `nvidia-smi` shows, but
closer to the actual requirements.
- We report the inference time as the overall time including data loading,
network forwarding and post processing.
- The training memory and time of 2x schedule is simply copied from 1x.
It should be very close to the actual memory and time.


## Baselines

We released RPN, Faster R-CNN and Mask R-CNN models in the first version. More models with different backbones will be added to the model zoo.

### RPN

| Backbone | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | AR1000 | Download |
|:--------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50-FPN | caffe   | 1x      | 4.5      | 0.379               | 14.4           | 58.2   | -        |
| R-50-FPN | pytorch | 1x      | 4.8      | 0.407               | 14.5           | 57.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_fpn_1x_20181010-4a9c0712.pth) \| [result](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/results/rpn_r50_fpn_1x_20181010_results.pkl.json) |
| R-50-FPN | pytorch | 2x      | 4.8      | 0.407               | 14.5           | 57.6   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_fpn_2x_20181010-88a4a471.pth) \| [result](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/results/rpn_r50_fpn_2x_20181010_results.pkl.json) |

### Faster R-CNN

| Backbone | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:--------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50-FPN | caffe   | 1x      | 4.9      | 0.525               | 10.0           | 36.7   | -        |
| R-50-FPN | pytorch | 1x      | 5.1      | 0.554               | 9.9            | 36.4   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth) \| [result](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/results/faster_rcnn_r50_fpn_1x_20181010_results.pkl.json) |
| R-50-FPN | pytorch | 2x      | 5.1      | 0.554               | 9.9            | 37.7   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth) \| [result](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/results/faster_rcnn_r50_fpn_2x_20181010_results.pkl.json) |

### Mask R-CNN

| Backbone | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
|:--------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
| R-50-FPN | caffe   | 1x      | 5.9      | 0.658               | 7.7            | 37.5   | 34.4    | -        |
| R-50-FPN | pytorch | 1x      | 5.8      | 0.690               | 7.7            | 37.3   | 34.2    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth) \| [result](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/results/mask_rcnn_r50_fpn_1x_20181010_results.pkl.json) |
| R-50-FPN | pytorch | 2x      | 5.8      | 0.690               | 7.7            | 38.6   | 35.1    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth) \| [result](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/results/mask_rcnn_r50_fpn_2x_20181010_results.pkl.json) |

### Fast R-CNN (with pre-computed proposals)

| Backbone | Style   | Type   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
|:--------:|:-------:|:------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
| R-50-FPN | caffe   | Faster | 1x      | 3.5      | 0.35                | 14.6           | 36.6   | -       | -        |
| R-50-FPN | pytorch | Faster | 1x      | 4.0      | 0.38                | 14.5           | 35.8   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_rcnn_r50_fpn_1x_20181010-08160859.pth) \| [result](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/results/fast_rcnn_r50_fpn_1x_20181010_results.pkl.json) |
| R-50-FPN | pytorch | Faster | 2x      | 4.0      | 0.38                | 14.5           | 37.1   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_rcnn_r50_fpn_2x_20181010-d263ada5.pth) \| [result](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/results/fast_rcnn_r50_fpn_2x_20181010_results.pkl.json) |
| R-50-FPN | caffe   | Mask   | 1x      | 5.4      | 0.47                | 10.7           | 37.3   | 34.5    | -        |
| R-50-FPN | pytorch | Mask   | 1x      | 5.3      | 0.50                | 10.6           | 36.8   | 34.1    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_mask_rcnn_r50_fpn_1x_20181010-e030a38f.pth) \| [result](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/results/fast_mask_rcnn_r50_fpn_1x_20181010_results.pkl.json) |
| R-50-FPN | pytorch | Mask   | 2x      | 5.3      | 0.50                | 10.6           | 37.9   | 34.8    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fast_mask_rcnn_r50_fpn_2x_20181010-5048cb03.pth) \| [result](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/results/fast_mask_rcnn_r50_fpn_2x_20181010_results.pkl.json) |

### RetinaNet (coming soon)

| Backbone | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:--------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50-FPN | caffe   | 1x      |          |                     |                |        |          |
| R-50-FPN | pytorch | 1x      |          |                     |                |        |          |
| R-50-FPN | pytorch | 2x      |          |                     |                |        |          |


## Comparison with Detectron

We compare mmdetection with [Detectron](https://github.com/facebookresearch/Detectron)
and [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch),
a third-party port of Detectron to Pytorch. The backbone used is R-50-FPN.

In general, mmdetection has 3 advantages over Detectron.

- **Higher performance** (especially in terms of mask AP)
- **Faster training speed**
- **Memory efficient**

### Performance

Detectron and Detectron.pytorch use caffe-style ResNet as the backbone.
In order to utilize the PyTorch model zoo, we use pytorch-style ResNet in our experiments.

In the meanwhile, we train models with caffe-style ResNet in 1x experiments for comparison.
We find that pytorch-style ResNet usually converges slower than caffe-style ResNet,
thus leading to slightly lower results in 1x schedule, but the final results
of 2x schedule is higher.

We report results using both caffe-style (weights converted from
[here](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#imagenet-pretrained-models))
and pytorch-style (weights from the official model zoo) ResNet backbone,
indicated as *pytorch-style results* / *caffe-style results*.

<table>
  <tr>
    <th>Type</th>
    <th>Lr schd</th>
    <th>Detectron</th>
    <th>Detectron.pytorch</th>
    <th>mmdetection</th>
  </tr>
  <tr>
    <td rowspan="2">RPN</td>
    <td>1x</td>
    <td>57.2</td>
    <td>-</td>
    <td>57.1 / 58.2</td>
  </tr>
  <tr>
    <td>2x</td>
    <td>-</td>
    <td>-</td>
    <td>57.6 / -</td>
  </tr>
  <tr>
    <td rowspan="2">Faster R-CNN</td>
    <td>1x</td>
    <td>36.7</td>
    <td>37.1</td>
    <td>36.4 / 36.7</td>
  </tr>
  <tr>
    <td>2x</td>
    <td>37.9</td>
    <td>-</td>
    <td>37.7 / -</td>
  </tr>
  <tr>
    <td rowspan="2">Mask R-CNN</td>
    <td>1x</td>
    <td>37.7 &amp; 33.9</td>
    <td>37.7 &amp; 33.7</td>
    <td>37.3 &amp; 34.2 / 37.5 &amp; 34.4</td>
  </tr>
  <tr>
    <td>2x</td>
    <td>38.6 &amp; 34.5</td>
    <td>-</td>
    <td>38.6 &amp; 35.1 / -</td>
  </tr>
  <tr>
    <td rowspan="2">Fast R-CNN</td>
    <td>1x</td>
    <td>36.4</td>
    <td>-</td>
    <td>35.8 / 36.6</td>
  </tr>
  <tr>
    <td>2x</td>
    <td>36.8</td>
    <td>-</td>
    <td>37.1 / -</td>
  </tr>
  <tr>
    <td rowspan="2">Fast R-CNN (w/mask)</td>
    <td>1x</td>
    <td>37.3 &amp; 33.7</td>
    <td>-</td>
    <td>36.8 &amp; 34.1 / 37.3 &amp; 34.5</td>
  </tr>
  <tr>
    <td>2x</td>
    <td>37.7 &amp; 34.0</td>
    <td>-</td>
    <td>37.9 &amp; 34.8 / -</td>
  </tr>
</table>

### Training Speed

The training speed is measure with s/iter. The lower, the better.

<table>
  <tr>
    <th>Type</th>
    <th>Detectron (P100<sup>1</sup>)</th>
    <th>Detectron.pytorch (XP<sup>2</sup>)</th>
    <th>mmdetection<sup>3</sup> (V100<sup>4</sup> / XP)</th>
  </tr>
  <tr>
    <td>RPN</td>
    <td>0.416</td>
    <td>-</td>
    <td>0.407 / 0.413</td>
  </tr>
  <tr>
    <td>Faster R-CNN</td>
    <td>0.544</td>
    <td>1.015</td>
    <td>0.554 / 0.579</td>
  </tr>
  <tr>
    <td>Mask R-CNN</td>
    <td>0.889</td>
    <td>1.435</td>
    <td>0.690 / 0.732</td>
  </tr>
  <tr>
    <td>Fast R-CNN</td>
    <td>0.285</td>
    <td>-</td>
    <td>0.375 / 0.398</td>
  </tr>
  <tr>
    <td>Fast R-CNN (w/mask)</td>
    <td>0.377</td>
    <td>-</td>
    <td>0.504 / 0.574</td>
  </tr>
</table>

\*1. Detectron reports the speed on Facebook's Big Basin servers (P100),
on our V100 servers it is slower so we use the official reported values.

\*2. Detectron.pytorch does not report the runtime and we encountered some issue to
run it on V100, so we report the speed on TITAN XP.

\*3. The speed of pytorch-style ResNet is approximately 5% slower than caffe-style,
and we report the pytorch-style results here.

\*4. We also run the models on a DGX-1 server (P100) and the speed is almost the same as our V100 servers.

### Inference Speed

The inference speed is measured with fps (img/s) on a single GPU. The higher, the better.

<table>
  <tr>
    <th>Type</th>
    <th>Detectron (P100)</th>
    <th>Detectron.pytorch (XP)</th>
    <th>mmdetection (V100 / XP)</th>
  </tr>
  <tr>
    <td>RPN</td>
    <td>12.5</td>
    <td>-</td>
    <td>14.5 / 15.4</td>
  </tr>
  <tr>
    <td>Faster R-CNN</td>
    <td>10.3</td>
    <td></td>
    <td>9.9 / 9.8</td>
  </tr>
  <tr>
    <td>Mask R-CNN</td>
    <td>8.5</td>
    <td></td>
    <td>7.7 / 7.4</td>
  </tr>
  <tr>
    <td>Fast R-CNN</td>
    <td>12.5</td>
    <td></td>
    <td>14.5 / 14.1</td>
  </tr>
  <tr>
    <td>Fast R-CNN (w/mask)</td>
    <td>9.9</td>
    <td></td>
    <td>10.6 / 10.3</td>
  </tr>
</table>

### Training memory

We perform various tests and there is no doubt that mmdetection is more memory
efficient than Detectron, and the main cause is the deep learning framework itself, not our efforts.
Besides, Caffe2 and PyTorch have different apis to obtain memory usage
whose implementation is not exactly the same.

`nvidia-smi` shows a larger memory usage for both detectron and mmdetection, e.g.,
we observe a much higher memory usage when we train Mask R-CNN with 2 images per GPU using detectron (10.6G) and mmdetection (9.3G), which is obviously more than actually required.

> With mmdetection, we can train R-50 FPN Mask R-CNN with **4** images per GPU (TITAN XP, 12G),
which is a promising result.
