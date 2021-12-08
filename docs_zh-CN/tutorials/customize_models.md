# 教程 4: 自定义模型

我们基本可以把模型的各个组件分为五类：

- 主干网络 (backbone)：通常是一个用来提取特征图 (feature map) 的全连接网络 (FCN network)，例如：ResNet, MobileNet。
- 颈部 (neck)：主干网络和头部之间的部分，例如：FPN, PAFPN。
- 头部 (head)：用于具体任务的模块，例如：边界框预测和掩膜预测。
- roi提取器 (roi extractor)：从特征图中提取roi特征，例如：RoI Align。
- 损失 (loss)：在头部组件中用于计算损失的部分，例如：FocalLoss, L1Loss, GHMLoss.

## 开发新的组件

### 添加一个新的主干网络

这里，我们以MobileNet为例来展示如何开发新组件。

#### 1. 定义一个新的主干网络（以MobileNet为例）

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




































