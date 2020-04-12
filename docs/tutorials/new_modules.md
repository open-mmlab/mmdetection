# Tutorials 1: Adding New Modules

## Customize optimizer

An example of customized optimizer `CopyOfSGD` is defined in `mmdet/core/optimizer/copy_of_sgd.py`.
More generally, a customized optimizer could be defined as following.

In `mmdet/core/optimizer/my_optimizer.py`:

```python
from .registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module
class MyOptimizer(Optimizer):

```

In `mmdet/core/optimizer/__init__.py`:

```python
from .my_optimizer import MyOptimizer
```

Then you can use `MyOptimizer` in `optimizer` field of config files.

## Develop new components

We basically categorize model components into 4 types.

- backbone: usually an FCN network to extract feature maps, e.g., ResNet, MobileNet.
- neck: the component between backbones and heads, e.g., FPN, PAFPN.
- head: the component for specific tasks, e.g., bbox prediction and mask prediction.
- roi extractor: the part for extracting RoI features from feature maps, e.g., RoI Align.

Here we show how to develop new components with an example of MobileNet.

1. Create a new file `mmdet/models/backbones/mobilenet.py`.

```python
import torch.nn as nn

from ..registry import BACKBONES


@BACKBONES.register_module
class MobileNet(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tuple
        pass

    def init_weights(self, pretrained=None):
        pass
```

2. Import the module in `mmdet/models/backbones/__init__.py`.

```python
from .mobilenet import MobileNet
```

3. Use it in your config file.

```python
model = dict(
    ...
    backbone=dict(
        type='MobileNet',
        arg1=xxx,
        arg2=xxx),
    ...
```
