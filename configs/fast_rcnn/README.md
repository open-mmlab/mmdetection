# Fast R-CNN

> [Fast R-CNN](https://arxiv.org/abs/1504.08083)

<!-- [ALGORITHM] -->

## Abstract

This paper proposes a Fast Region-based Convolutional Network method (Fast R-CNN) for object detection. Fast R-CNN builds on previous work to efficiently classify object proposals using deep convolutional networks. Compared to previous work, Fast R-CNN employs several innovations to improve training and testing speed while also increasing detection accuracy. Fast R-CNN trains the very deep VGG16 network 9x faster than R-CNN, is 213x faster at test-time, and achieves a higher mAP on PASCAL VOC 2012. Compared to SPPnet, Fast R-CNN trains VGG16 3x faster, tests 10x faster, and is more accurate.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143882189-6258c05c-f2a1-4320-9282-7e2f2d502eb2.png"/>
</div>

## Introduction

Before training the Fast R-CNN, users should first train an [RPN](../rpn/README.md), and use the RPN to extract the region proposals.

- Firstly, extract the region proposals of the val set by this command as below:

```bash
./tools/dist_test.sh \
    configs/rpn_r50_fpn_1x_coco.py \
    checkpoints/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth \
    8 \
    --out proposals/rpn_r50_fpn_1x_val2017.pkl
```

- Then, change the `ann_file` and `img_prefix` of `data.test` in the RPN config to train set as below:

```python
data = dict(
    test=dict(
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/'))
```

- Extract the region proposals of the train set by this command as below:

```bash
./tools/dist_test.sh \
    configs/rpn_r50_fpn_1x_coco.py \
    checkpoints/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth \
    8 \
    --out proposals/rpn_r50_fpn_1x_train2017.pkl
```

- Modify the path of `proposal_file` in Fast R-CNN config as below:

```python
data = dict(
    train=dict(
        proposal_file='proposals/rpn_r50_fpn_1x_train2017.pkl'),
    val=dict(
        proposal_file='proposals/rpn_r50_fpn_1x_val2017.pkl'),
    test=dict(
        proposal_file='proposals/rpn_r50_fpn_1x_val2017.pkl'))
```

Finally, users can start training the Fast R-CNN.

## Results and Models

## Citation

```latex
@inproceedings{girshick2015fast,
  title={Fast r-cnn},
  author={Girshick, Ross},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2015}
}
```
