# 教程 10: 权重初始化

在训练过程中，适当的初始化策略有利于加快训练速度或获得更⾼的性能。 [MMCV](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/weight_init.py) 提供了一些常⽤的初始化模块的⽅法，如 `nn.Conv2d`。 MMdetection 中的模型初始化主要使⽤ `init_cfg`。⽤⼾可以通过以下两个步骤来初始化模型：

1. 在 `model_cfg` 中为模型或其组件定义 `init_cfg`，但⼦组件的 `init_cfg` 优先级更⾼，会覆盖⽗模块的 `init_cfg` 。
2. 像往常一样构建模型，然后显式调⽤ `model.init_weights()` ⽅法，此时模型参数将会被按照配置文件写法进行初始化。

MMdetection 初始化工作流的高层 API 调用流程是：

model_cfg(init_cfg) -> build_from_cfg -> model -> init_weight() -> initialize(self, self.init_cfg) -> children's init_weight()

### 描述

它的数据类型是 dict 或者 list\[dict\]，包含了下列键值:

- `type` (str)，包含 `INTIALIZERS` 中的初始化器名称，后面跟着初始化器的参数。
- `layer`（str 或 list\[str\]），包含 Pytorch 或 MMCV 中基本层的名称，以及将被初始化的可学习参数，例如 `'Conv2d'`，`'DeformConv2d'`。
- `override` (dict 或 list\[dict\])，包含不继承⾃ `BaseModule` 且其初始化配置与 `layer` 键中的其他层不同的⼦模块。 `type` 中定义的初始化器将适⽤于 `layer` 中定义的所有层，因此如果⼦模块不是 `BaseModule` 的派⽣类但可以与 `layer` 中的层相同的⽅式初始化，则不需要使⽤ `override`。`override` 包含了：
  - `type` 后跟初始化器的参数；
  - `name` 用以指⽰将被初始化的⼦模块。

### 初始化参数

从 `mmcv.runner.BaseModule` 或 `mmdet.models` 继承一个新模型。这里我们用 FooModel 来举个例子。

```python
import torch.nn as nn
from mmcv.runner import BaseModule

class FooModel(BaseModule)
	def __init__(self,
                 arg1,
                 arg2,
                 init_cfg=None):
    	super(FooModel, self).__init__(init_cfg)
		...
```

- 直接在代码中使⽤ `init_cfg` 初始化模型

  ```python
  import torch.nn as nn
  from mmcv.runner import BaseModule
  # or directly inherit mmdet models

  class FooModel(BaseModule)
  	def __init__(self,
                  arg1,
                  arg2,
                  init_cfg=XXX):
    		super(FooModel, self).__init__(init_cfg)
    	    ...
  ```

- 在 `mmcv.Sequential` 或 `mmcv.ModuleList` 代码中直接使⽤ `init_cfg` 初始化模型

  ```python
  from mmcv.runner import BaseModule, ModuleList

  class FooModel(BaseModule)
  	def __init__(self,
                	arg1,
                	arg2,
                	init_cfg=None):
    		super(FooModel, self).__init__(init_cfg)
        	...
        	self.conv1 = ModuleList(init_cfg=XXX)
  ```

- 使⽤配置⽂件中的 `init_cfg` 初始化模型

  ```python
  model = dict(
  	...
    	model = dict(
        	type='FooModel',
        	arg1=XXX,
        	arg2=XXX,
        	init_cfg=XXX),
            ...
  ```

### init_cfg 的使用

1. 用 `layer` 键初始化模型

   如果我们只定义了 `layer`, 它只会在 `layer` 键中初始化网络层。

   注意： `layer` 键对应的值是 Pytorch 的带有 weights 和 bias 属性的类名（因此不⽀持 `MultiheadAttention` 层）。

- 定义⽤于初始化具有相同配置的模块的 `layer` 键。

  ```python
  init_cfg = dict(type='Constant', layer=['Conv1d', 'Conv2d', 'Linear'], val=1)
  # ⽤相同的配置初始化整个模块
  ```

- 定义⽤于初始化具有不同配置的层的 `layer` 键。

  ```python
  init_cfg = [dict(type='Constant', layer='Conv1d', val=1),
              dict(type='Constant', layer='Conv2d', val=2),
              dict(type='Constant', layer='Linear', val=3)]
  # nn.Conv1d 将被初始化为 dict(type='Constant', val=1)
  # nn.Conv2d 将被初始化为 dict(type='Constant', val=2)
  # nn.Linear 将被初始化为 dict(type='Constant', val=3)
  ```

2. 使⽤ `override` 键初始化模型

- 当使⽤属性名初始化某些特定部分时，我们可以使⽤ `override` 键， `override` 中的值将忽略 init_cfg 中的值。

  ```python
  # layers：
  # self.feat = nn.Conv1d(3, 1, 3)
  # self.reg = nn.Conv2d(3, 3, 3)
  # self.cls = nn.Linear(1,2)

  init_cfg = dict(type='Constant',
                  layer=['Conv1d','Conv2d'], val=1, bias=2,
                  override=dict(type='Constant', name='reg', val=3, bias=4))
  # self.feat and self.cls 将被初始化为 dict(type='Constant', val=1, bias=2)
  # 叫 'reg' 的模块将被初始化为 dict(type='Constant', val=3, bias=4)
  ```

- 如果 init_cfg 中的 `layer` 为 None，则只会初始化 override 中有 name 的⼦模块，⽽ override 中的 type 和其他参数可以省略。

  ```python
  # layers：
  # self.feat = nn.Conv1d(3, 1, 3)
  # self.reg = nn.Conv2d(3, 3, 3)
  # self.cls = nn.Linear(1,2)

  init_cfg = dict(type='Constant', val=1, bias=2, 	override=dict(name='reg'))

  # self.feat and self.cls 将被 Pytorch 初始化
  # 叫 'reg' 的模块将被 dict(type='Constant', val=1, bias=2) 初始化
  ```

- 如果我们不定义 `layer` 或 `override` 键，它不会初始化任何东西。

- 无效的使用

  ```python
  # override 没有 name 键的话是无效的
  init_cfg = dict(type='Constant', layer=['Conv1d','Conv2d'], val=1, bias=2,
              	override=dict(type='Constant', val=3, bias=4))

  # override 有 name 键和其他参数但是没有 type 键也是无效的
  init_cfg = dict(type='Constant', layer=['Conv1d','Conv2d'], val=1, bias=2,
                  override=dict(name='reg', val=3, bias=4))
  ```

3. 使⽤预训练模型初始化模型

   ```python
   init_cfg = dict(type='Pretrained',
                checkpoint='torchvision://resnet50')
   ```

更多细节可以参考 [MMCV](https://mmcv.readthedocs.io/en/latest/cnn.html#weight-initialization) 的文档和 MMCV [PR #780](https://github.com/open-mmlab/mmcv/pull/780)
