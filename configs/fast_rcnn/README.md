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

- Firstly,
  change the type of `test_evaluator` to `DumpProposals` in the RPN config to get the extract the region proposals of the val set as below:

```python
test_evaluator = dict(
    _delete_=True,
    type='DumpProposals',
    output_dir='data/coco/proposals/',
    proposals_file='rpn_r50_fpn_1x_val2017.pkl'),
```

- Then, change the `ann_file` and `data_prefix.img` of `test_dataloader.dataset` in the RPN config to train set as below:

```python
test_dataloader = dict(
    dataset=dict(
        ann_file='data/coco/annotations/instances_train2017.json',
        data_prefix=dict(img='val2017/')))
test_evaluator = dict(
    _delete_=True,
    type='DumpProposals',
    output_dir='data/coco/proposals/',
    proposals_file='rpn_r50_fpn_1x_train2017.pkl'),
```

- Modify the path of `proposal_file` in Fast R-CNN config and using `ProposalBroadcaster` in the pipeline as below:

```python
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadProposals', num_max_proposals=2000),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='ProposalBroadcaster',
        transforms=[
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
        ]),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(
        type='ProposalBroadcaster',
        transforms=[
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    dataset=dict(
        proposal_file='proposals/rpn_r50_fpn_1x_train2017.pkl',
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        proposal_file='proposals/rpn_r50_fpn_1x_val2017.pkl',
        pipeline=test_pipeline))
test_dataloader = val_dataloader
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
