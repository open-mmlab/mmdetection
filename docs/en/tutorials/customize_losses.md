# Tutorial 6: Customize Losses

MMDetection provides users with different loss functions. But the default configuration may be not applicable for different datasets or models, so users may want to modify a specific loss to adapt the new situation.

This tutorial first elaborate the computation pipeline of losses, then give some instructions about how to modify each step. The modification can be categorized as tweaking and weighting.

## Computation pipeline of a loss

Given the input prediction and target, as well as the weights, a loss function maps the input tensor to the final loss scalar. The mapping can be divided into five steps:

1. Set the sampling method to sample positive and negative samples.

2. Get **element-wise** or **sample-wise** loss by the loss kernel function.

3. Weighting the loss with a weight tensor **element-wisely**.

4. Reduce the loss tensor to a **scalar**.

5. Weighting the loss with a **scalar**.

## Set sampling method (step 1)

For some loss functions, sampling strategies are needed to avoid imbalance between positive and negative samples.

For example, when using `CrossEntropyLoss` in RPN head, we need to set `RandomSampler` in `train_cfg`

```python
train_cfg=dict(
    rpn=dict(
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False))
```

For some other losses which have positive and negative sample balance mechanism such as Focal Loss, GHMC, and QualityFocalLoss, the sampler is no more necessary.

## Tweaking loss

Tweaking a loss is more related with step 2, 4, 5, and most modifications can be specified in the config.
Here we take [Focal Loss (FL)](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/focal_loss.py) as an example.
The following code sniper are the construction method and config of FL respectively, they are actually one to one correspondence.

```python
@LOSSES.register_module()
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
```

```python
loss_cls=dict(
    type='FocalLoss',
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.25,
    loss_weight=1.0)
```

### Tweaking hyper-parameters (step 2)

`gamma` and `beta` are two hyper-parameters in the Focal Loss. Say if we want to change the value of `gamma` to be 1.5 and `alpha` to be 0.5, then we can specify them in the config as follows:

```python
loss_cls=dict(
    type='FocalLoss',
    use_sigmoid=True,
    gamma=1.5,
    alpha=0.5,
    loss_weight=1.0)
```

### Tweaking the way of reduction (step 3)

The default way of reduction is `mean` for FL. Say if we want to change the reduction from `mean` to `sum`, we can specify it in the config as follows:

```python
loss_cls=dict(
    type='FocalLoss',
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.25,
    loss_weight=1.0,
    reduction='sum')
```

### Tweaking loss weight (step 5)

The loss weight here is a scalar which controls the weight of different losses in multi-task learning, e.g. classification loss and regression loss. Say if we want to change to loss weight of classification loss to be 0.5, we can specify it in the config as follows:

```python
loss_cls=dict(
    type='FocalLoss',
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.25,
    loss_weight=0.5)
```

## Weighting loss (step 3)

Weighting loss means we re-weight the loss element-wisely. To be more specific, we multiply the loss tensor with a weight tensor which has the same shape. As a result, different entries of the loss can be scaled differently, and so called element-wisely.
The loss weight varies across different models and highly context related, but overall there are two kinds of loss weights, `label_weights` for classification loss and `bbox_weights` for bbox regression loss. You can find them in the `get_target` method of the corresponding head. Here we take [ATSSHead](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/atss_head.py#L530) as an example, which inherit [AnchorHead](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/anchor_head.py) but overwrite its `get_targets` method which yields different `label_weights` and `bbox_weights`.

```
class ATSSHead(AnchorHead):

    ...

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
```
