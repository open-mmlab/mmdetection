# 教程 6: 自定义损失函数

MMDetection 为用户提供了不同的损失函数。但是默认的配置可能无法适应不同的数据和模型，所以用户可能会希望修改某一个损失函数来适应新的情况。

本教程首先详细的解释计算损失的过程然后给出一些关于如何修改每一个步骤的指导。对损失的修改可以被分为微调和加权。

## 一个损失的计算过程

给定输入（包括预测和目标，以及权重），损失函数会把输入的张量映射到最后的损失标量。映射过程可以分为下面五个步骤：

1. 设置采样方法为对正负样本进行采样。

2. 通过损失核函数获取**元素**或者**样本**损失。

3. 通过权重张量来给损失**逐元素**权重。

4. 把损失张量归纳为一个**标量**。

5. 用一个**张量**给当前损失一个权重。

## 设置采样方法（步骤 1）

对于一些损失函数，需要采样策略来避免正负样本之间的不平衡。

例如，在RPN head中使用`CrossEntropyLoss`时，我们需要在`train_cfg`中设置`RandomSampler`

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

对于其他一些具有正负样本平衡机制的损失，例如 Focal Loss、GHMC 和 QualityFocalLoss，不再需要进行采样。

## 微调损失

微调一个损失主要与步骤 2，4，5 有关，大部分的修改可以在配置文件中指定。这里我们用 [Focal Loss (FL)](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/focal_loss.py) 作为例子。
下面的代码分别是构建 FL 的方法和它的配置文件，他们是一一对应的。

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

### 微调超参数（步骤2）

`gamma` 和 `beta` 是 Focal Loss 中的两个超参数。如果我们想把 `gamma` 的值设为 1.5，把 `alpha` 的值设为 0.5，我们可以在配置文件中按照如下指定：

```python
loss_cls=dict(
    type='FocalLoss',
    use_sigmoid=True,
    gamma=1.5,
    alpha=0.5,
    loss_weight=1.0)
```

### 微调归纳方式（步骤4）

Focal Loss 默认的归纳方式是 `mean`。如果我们想把归纳方式从 `mean` 改成 `sum`，我们可以在配置文件中按照如下指定：

```python
loss_cls=dict(
    type='FocalLoss',
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.25,
    loss_weight=1.0,
    reduction='sum')
```

### 微调损失权重（步骤5）

这里的损失权重是一个标量，他用来控制多任务学习中不同损失的重要程度，例如，分类损失和回归损失。如果我们想把分类损失的权重设为 0.5，我们可以在配置文件中如下指定：

```python
loss_cls=dict(
    type='FocalLoss',
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.25,
    loss_weight=0.5)
```

## 加权损失（步骤3）

加权损失就是我们逐元素修改损失权重。更具体来说，我们给损失张量乘以一个与他有相同形状的权重张量。所以，损失中不同的元素可以被赋予不同的比例，所以这里叫做逐元素。损失的权重在不同模型中变化很大，而且与上下文相关，但是总的来说主要有两种损失权重：分类损失的 `label_weights` 和边界框的 `bbox_weights`。你可以在相应的头中的 `get_target` 方法中找到他们。这里我们使用 [ATSSHead](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/atss_head.py#L530) 作为一个例子。它继承了 [AnchorHead](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/anchor_head.py)，但是我们重写它的
`get_targets` 方法来产生不同的 `label_weights` 和 `bbox_weights`。

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
