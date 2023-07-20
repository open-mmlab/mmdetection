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
The region proposals can be obtained by setting `DumpProposals` pseudo metric. The dumped results is a `dict(file_name: pred_instance)`.
The `pred_instance` is an `InstanceData` containing the sorted boxes and scores predicted by RPN. We provide example of dumping proposals in [RPN config](../rpn/rpn_r50_fpn_1x_coco.py).

- First, it should be obtained the region proposals in both training and validation (or testing) set.
  change the type of `test_evaluator` to `DumpProposals` in the RPN config to get the region proposals as below:

  The config of get training image region proposals can be set as below:

  ```python
  # For training set
  val_dataloader = dict(
      dataset=dict(
          ann_file='data/coco/annotations/instances_train2017.json',
          data_prefix=dict(img='val2017/')))
  val_dataloader = dict(
      _delete_=True,
      type='DumpProposals',
      output_dir='data/coco/proposals/',
      proposals_file='rpn_r50_fpn_1x_train2017.pkl')
  test_dataloader = val_dataloader
  test_evaluator = val_dataloader
  ```

  The config of get validation image region proposals can be set as below:

  ```python
  # For validation set
  val_dataloader = dict(
    _delete_=True,
    type='DumpProposals',
    output_dir='data/coco/proposals/',
    proposals_file='rpn_r50_fpn_1x_val2017.pkl')
  test_evaluator = val_dataloader
  ```

  Extract the region proposals command can be set as below:

  ```bash
  ./tools/dist_test.sh \
      configs/rpn_r50_fpn_1x_coco.py \
      checkpoints/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth \
      8
  ```

  Users can refer to [test tutorial](https://mmdetection.readthedocs.io/en/latest/user_guides/test.html) for more details.

- Then, modify the path of `proposal_file` in the dataset and using `ProposalBroadcaster` to process both ground truth bounding boxes and region proposals in pipelines.
  An example of Fast R-CNN important setting can be seen as below:

  ```python
  train_pipeline = [
      dict(
          type='LoadImageFromFile',
          backend_args={{_base_.backend_args}}),
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
          backend_args={{_base_.backend_args}}),
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

- Finally, users can start training the Fast R-CNN.

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
