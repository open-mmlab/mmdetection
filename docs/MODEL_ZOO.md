# Benchmark and Model Zoo

## Environment

### Hardware

- 8 NVIDIA Tesla V100 GPUs
- Intel Xeon 4114 CPU @ 2.20GHz

### Software environment

- Python 3.6 / 3.7
- PyTorch 1.1
- CUDA 9.0.176
- CUDNN 7.0.4
- NCCL 2.1.15

## Mirror sites

We use AWS as the main site to host our model zoo, and maintain a mirror on aliyun.
You can replace `https://s3.ap-northeast-2.amazonaws.com/open-mmlab` with `https://open-mmlab.oss-cn-beijing.aliyuncs.com` in model urls.

## Common settings

- All FPN baselines and RPN-C4 baselines were trained using 8 GPU with a batch size of 16 (2 images per GPU). Other C4 baselines were trained using 8 GPU with a batch size of 8 (1 image per GPU).
- All models were trained on `coco_2017_train`, and tested on the `coco_2017_val`.
- We use distributed training and BN layer stats are fixed.
- We adopt the same training schedules as Detectron. 1x indicates 12 epochs and 2x indicates 24 epochs, which corresponds to slightly less iterations than Detectron and the difference can be ignored.
- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo.
- For fair comparison with other codebases, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 8 GPUs. Note that this value is usually less than what `nvidia-smi` shows.
- We report the inference time as the overall time including data loading, network forwarding and post processing.


## Baselines

More models with different backbones will be added to the model zoo.

### RPN

### Faster R-CNN

Please refer to [Faster R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn) for details.

### Mask R-CNN

Please refer to [Mask R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn) for details.

### Fast R-CNN (with pre-computed proposals)

Please refer to [Fast R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/fast_rcnn) for details.

### RetinaNet

Please refer to [RetinaNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet) for details.

### Cascade R-CNN

Please refer to [Cascade R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/cascade_rcnn) for details.

### Cascade Mask R-CNN

Please refer to [Cascade Mask R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/cascade_rcnn) for details.

### Hybrid Task Cascade (HTC)

Please refer to [HTC](https://github.com/open-mmlab/mmdetection/blob/master/configs/htc) for details.

### SSD

Please refer to [SSD](https://github.com/open-mmlab/mmdetection/blob/master/configs/ssd) for details.

### Group Normalization (GN)

Please refer to [Group Normalization](https://github.com/open-mmlab/mmdetection/blob/master/configs/gn) for details.

### Weight Standardization

Please refer to [Weight Standardization](https://github.com/open-mmlab/mmdetection/blob/master/configs/gn+ws) for details.

### Deformable Convolution v2

Please refer to [Deformable Convolutional Networks](https://github.com/open-mmlab/mmdetection/blob/master/configs/dcn) for details.

### CARAFE: Content-Aware ReAssembly of FEatures
Please refer to [CARAFE](https://github.com/open-mmlab/mmdetection/blob/master/configs/carafe) for details.

### Instaboost

Please refer to [Instaboost](https://github.com/open-mmlab/mmdetection/blob/master/configs/instaboost) for details.

### Libra R-CNN

Please refer to [Libra R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/libra_rcnn) for details.

### Guided Anchoring

Please refer to [Guided Anchoring](https://github.com/open-mmlab/mmdetection/blob/master/configs/guided_anchoring) for details.

### FCOS

Please refer to [FCOS](https://github.com/open-mmlab/mmdetection/blob/master/configs/fcos) for details.

### FoveaBox

Please refer to [FoveaBox](https://github.com/open-mmlab/mmdetection/blob/master/configs/foveabox) for details.

### RepPoints

Please refer to [RepPoints](https://github.com/open-mmlab/mmdetection/blob/master/configs/reppoints) for details.

### FreeAnchor

Please refer to [FreeAnchor](https://github.com/open-mmlab/mmdetection/blob/master/configs/free_anchor) for details.

### Grid R-CNN (plus)

Please refer to [Grid R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/grid_rcnn) for details.

### GHM

Please refer to [GHM](https://github.com/open-mmlab/mmdetection/blob/master/configs/ghm) for details.

### GCNet

Please refer to [GCNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/gcnet) for details.

### HRNet
Please refer to [HRNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/hrnet) for details.

### Mask Scoring R-CNN

Please refer to [Mask Scoring R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/ms_rcnn) for details.

### Train from Scratch

Please refer to [Rethinking ImageNet Pre-training](https://github.com/open-mmlab/mmdetection/blob/master/configs/scratch) for details.

### NAS-FPN
Please refer to [NAS-FPN](https://github.com/open-mmlab/mmdetection/blob/master/configs/nas_fpn) for details.

### ATSS
Please refer to [ATSS](https://github.com/open-mmlab/mmdetection/blob/master/configs/atss) for details.

### Other datasets

We also benchmark some methods on [PASCAL VOC](https://github.com/open-mmlab/mmdetection/blob/master/configs/pascal_voc), [Cityscapes](https://github.com/open-mmlab/mmdetection/blob/master/configs/cityscapes) and [WIDER FACE](https://github.com/open-mmlab/mmdetection/blob/master/configs/wider_face).


## Comparison with Detectron and maskrcnn-benchmark

We compare mmdetection with [Detectron](https://github.com/facebookresearch/Detectron)
and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). The backbone used is R-50-FPN.

In general, mmdetection has 3 advantages over Detectron.

- **Higher performance** (especially in terms of mask AP)
- **Faster training speed**
- **Memory efficient**

### Performance

Detectron and maskrcnn-benchmark use caffe-style ResNet as the backbone.
We report results using both caffe-style (weights converted from
[here](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#imagenet-pretrained-models))
and pytorch-style (weights from the official model zoo) ResNet backbone,
indicated as *pytorch-style results* / *caffe-style results*.

We find that pytorch-style ResNet usually converges slower than caffe-style ResNet,
thus leading to slightly lower results in 1x schedule, but the final results
of 2x schedule is higher.

<table>
  <tr>
    <th>Type</th>
    <th>Lr schd</th>
    <th>Detectron</th>
    <th>maskrcnn-benchmark</th>
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
    <td>36.8</td>
    <td>36.4 / 36.6</td>
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
    <td>37.8 &amp; 34.2</td>
    <td>37.3 &amp; 34.2 / 37.4 &amp; 34.3</td>
  </tr>
  <tr>
    <td>2x</td>
    <td>38.6 &amp; 34.5</td>
    <td>-</td>
    <td>38.5 &amp; 35.1 / -</td>
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
    <th>maskrcnn-benchmark (V100)</th>
    <th>mmdetection (V100<sup>2</sup>)</th>
  </tr>
  <tr>
    <td>RPN</td>
    <td>0.416</td>
    <td>-</td>
    <td>0.253</td>
  </tr>
  <tr>
    <td>Faster R-CNN</td>
    <td>0.544</td>
    <td>0.353</td>
    <td>0.333</td>
  </tr>
  <tr>
    <td>Mask R-CNN</td>
    <td>0.889</td>
    <td>0.454</td>
    <td>0.430</td>
  </tr>
  <tr>
    <td>Fast R-CNN</td>
    <td>0.285</td>
    <td>-</td>
    <td>0.242</td>
  </tr>
  <tr>
    <td>Fast R-CNN (w/mask)</td>
    <td>0.377</td>
    <td>-</td>
    <td>0.328</td>
  </tr>
</table>

\*1. Facebook's Big Basin servers (P100/V100) is slightly faster than the servers we use. mmdetection can also run slightly faster on FB's servers.

\*2. For fair comparison, we list the caffe-style results here.


### Inference Speed

The inference speed is measured with fps (img/s) on a single GPU. The higher, the better.

<table>
  <tr>
    <th>Type</th>
    <th>Detectron (P100)</th>
    <th>maskrcnn-benchmark (V100)</th>
    <th>mmdetection (V100)</th>
  </tr>
  <tr>
    <td>RPN</td>
    <td>12.5</td>
    <td>-</td>
    <td>16.9</td>
  </tr>
  <tr>
    <td>Faster R-CNN</td>
    <td>10.3</td>
    <td>7.9</td>
    <td>13.5</td>
  </tr>
  <tr>
    <td>Mask R-CNN</td>
    <td>8.5</td>
    <td>7.7</td>
    <td>10.2</td>
  </tr>
  <tr>
    <td>Fast R-CNN</td>
    <td>12.5</td>
    <td>-</td>
    <td>18.4</td>
  </tr>
  <tr>
    <td>Fast R-CNN (w/mask)</td>
    <td>9.9</td>
    <td>-</td>
    <td>12.8</td>
  </tr>
</table>

### Training memory

<table>
  <tr>
    <th>Type</th>
    <th>Detectron</th>
    <th>maskrcnn-benchmark</th>
    <th>mmdetection</th>
  </tr>
  <tr>
    <td>RPN</td>
    <td>6.4</td>
    <td>-</td>
    <td>3.3</td>
  </tr>
  <tr>
    <td>Faster R-CNN</td>
    <td>7.2</td>
    <td>4.4</td>
    <td>3.6</td>
  </tr>
  <tr>
    <td>Mask R-CNN</td>
    <td>8.6</td>
    <td>5.2</td>
    <td>3.8</td>
  </tr>
  <tr>
    <td>Fast R-CNN</td>
    <td>6.0</td>
    <td>-</td>
    <td>3.3</td>
  </tr>
  <tr>
    <td>Fast R-CNN (w/mask)</td>
    <td>7.9</td>
    <td>-</td>
    <td>3.4</td>
  </tr>
</table>

There is no doubt that maskrcnn-benchmark and mmdetection is more memory efficient than Detectron,
and the main advantage is PyTorch itself. We also perform some memory optimizations to push it forward.

Note that Caffe2 and PyTorch have different apis to obtain memory usage with different implementations.
For all codebases, `nvidia-smi` shows a larger memory usage than the reported number in the above table.
