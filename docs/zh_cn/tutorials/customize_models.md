# 教程 4: 自定义模型

我们简单地把模型的各个组件分为五类：

- 主干网络 (backbone)：通常是一个用来提取特征图 (feature map) 的全卷积网络 (FCN network)，例如：ResNet, MobileNet。
- Neck：主干网络和 Head 之间的连接部分，例如：FPN, PAFPN。
- Head：用于具体任务的组件，例如：边界框预测和掩码预测。
- 区域提取器 (roi extractor)：从特征图中提取 RoI 特征，例如：RoI Align。
- 损失 (loss)：在 Head 组件中用于计算损失的部分，例如：FocalLoss, L1Loss, GHMLoss.

## 开发新的组件

### 添加一个新的主干网络

这里，我们以 MobileNet 为例来展示如何开发新组件。

#### 1. 定义一个新的主干网络（以 MobileNet 为例）

新建一个文件 `mmdet/models/backbones/mobilenet.py`

```python
import torch.nn as nn

from ..builder import BACKBONES


@BACKBONES.register_module()
class MobileNet(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tuple
        pass
```

#### 2. 导入该模块

你可以添加下述代码到 `mmdet/models/backbones/__init__.py`

```python
from .mobilenet import MobileNet
```

或添加：

```python
custom_imports = dict(
    imports=['mmdet.models.backbones.mobilenet'],
    allow_failed_imports=False)
```

到配置文件以避免原始代码被修改。

#### 3. 在你的配置文件中使用该主干网络

```python
model = dict(
    ...
    backbone=dict(
        type='MobileNet',
        arg1=xxx,
        arg2=xxx),
    ...
```

### 添加新的 Neck

#### 1. 定义一个 Neck（以 PAFPN 为例）

新建一个文件 `mmdet/models/necks/pafpn.py`

```python
from ..builder import NECKS

@NECKS.register_module()
class PAFPN(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                num_outs,
                start_level=0,
                end_level=-1,
                add_extra_convs=False):
        pass

    def forward(self, inputs):
        # implementation is ignored
        pass
```

#### 2. 导入该模块

你可以添加下述代码到 `mmdet/models/necks/__init__.py`

```python
from .pafpn import PAFPN
```

或添加：

```python
custom_imports = dict(
    imports=['mmdet.models.necks.pafpn.py'],
    allow_failed_imports=False)
```

到配置文件以避免原始代码被修改。

#### 3. 修改配置文件

```python
neck=dict(
    type='PAFPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5)
```

### 添加新的 Head

我们以 [Double Head R-CNN](https://arxiv.org/abs/1904.06493) 为例来展示如何添加一个新的 Head。

首先，添加一个新的 bbox head 到 `mmdet/models/roi_heads/bbox_heads/double_bbox_head.py`。
Double Head R-CNN 在目标检测上实现了一个新的 bbox head。为了实现 bbox head，我们需要使用如下的新模块中三个函数。

```python
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead

@HEADS.register_module()
class DoubleConvFCBBoxHead(BBoxHead):
    r"""Bbox head used in Double-Head R-CNN

                                      /-> cls
                  /-> shared convs ->
                                      \-> reg
    roi features
                                      /-> cls
                  \-> shared fc    ->
                                      \-> reg
    """  # noqa: W605

    def __init__(self,
                 num_convs=0,
                 num_fcs=0,
                 conv_out_channels=1024,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(DoubleConvFCBBoxHead, self).__init__(**kwargs)


    def forward(self, x_cls, x_reg):

```

然后，如有必要，实现一个新的 bbox head。我们打算从 `StandardRoIHead` 来继承新的 `DoubleHeadRoIHead`。我们可以发现 `StandardRoIHead` 已经实现了下述函数。

```python
import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class StandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head.
    """

    def init_assigner_sampler(self):

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):

    def init_mask_head(self, mask_roi_extractor, mask_head):


    def forward_dummy(self, x, proposals):


    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):

    def _bbox_forward(self, x, rois):

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):


    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""

```

Double Head 的修改主要在 bbox_forward 的逻辑中，且它从 `StandardRoIHead` 中继承了其他逻辑。在 `mmdet/models/roi_heads/double_roi_head.py` 中，我们用下述代码实现新的 bbox head：

```python
from ..builder import HEADS
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class DoubleHeadRoIHead(StandardRoIHead):
    """RoI head for Double Head RCNN

    https://arxiv.org/abs/1904.06493
    """

    def __init__(self, reg_roi_scale_factor, **kwargs):
        super(DoubleHeadRoIHead, self).__init__(**kwargs)
        self.reg_roi_scale_factor = reg_roi_scale_factor

    def _bbox_forward(self, x, rois):
        bbox_cls_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            roi_scale_factor=self.reg_roi_scale_factor)
        if self.with_shared_head:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_cls_feats)
        return bbox_results
```

最终，用户需要把该模块添加到 `mmdet/models/bbox_heads/__init__.py` 和 `mmdet/models/roi_heads/__init__.py` 以使相关的注册表可以找到并加载他们。

或者，用户可以添加：

```python
custom_imports=dict(
    imports=['mmdet.models.roi_heads.double_roi_head', 'mmdet.models.bbox_heads.double_bbox_head'])
```

到配置文件并实现相同的目的。

Double Head R-CNN 的配置文件如下：

```python
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    roi_head=dict(
        type='DoubleHeadRoIHead',
        reg_roi_scale_factor=1.3,
        bbox_head=dict(
            _delete_=True,
            type='DoubleConvFCBBoxHead',
            num_convs=4,
            num_fcs=2,
            in_channels=256,
            conv_out_channels=1024,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0))))

```

从 MMDetection 2.0 版本起，配置系统支持继承配置以使用户可以专注于修改。
Double Head R-CNN 主要使用了一个新的 DoubleHeadRoIHead 和一个新的 `DoubleConvFCBBoxHead`，参数需要根据每个模块的 `__init__` 函数来设置。

### 添加新的损失

假设你想添加一个新的损失 `MyLoss` 用于边界框回归。
为了添加一个新的损失函数，用户需要在 `mmdet/models/losses/my_loss.py` 中实现。
装饰器 `weighted_loss` 可以使损失每个部分加权。

```python
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss

@weighted_loss
def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss

@LOSSES.register_module()
class MyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * my_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox
```

然后，用户需要把它加到 `mmdet/models/losses/__init__.py`。

```python
from .my_loss import MyLoss, my_loss

```

或者，你可以添加：

```python
custom_imports=dict(
    imports=['mmdet.models.losses.my_loss'])
```

到配置文件来实现相同的目的。

如使用，请修改 `loss_xxx` 字段。
因为 MyLoss 是用于回归的，你需要在 Head 中修改 `loss_xxx` 字段。

```python
loss_bbox=dict(type='MyLoss', loss_weight=1.0))
```
