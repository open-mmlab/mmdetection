# Tutorial 10: Weight initialization

During training, a proper initialization strategy is beneficial to speeding up the training or obtaining a higher performance. [MMCV](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/weight_init.py) provide some commonly used methods for initializing modules like `nn.Conv2d`. Model initialization in MMdetection mainly uses `init_cfg`. Users can initialize models with following two steps:

1. Define `init_cfg` for a model or its components in `model_cfg`,  but `init_cfg` of children components have higher priority and will override `init_cfg` of parents modules.
2. Build model as usual, but call `model.init_weights()` method explicitly, and model parameters will be initialized as configuration.

The high-level workflow of initialization in MMdetection is :

model_cfg(init_cfg) -> build_from_cfg -> model -> init_weight() -> initialize(self, self.init_cfg) -> children's init_weight()

### Description

It is dict or list[dict], and contains the following keys and values:

- `type` (str), containing the initializer name in `INTIALIZERS`, and followed by arguments of the initializer.
- `layer` (str or list[str]), containing the names of basiclayers in Pytorch or MMCV with learnable parameters that will be initialized, e.g. `'Conv2d'`,`'DeformConv2d'`.
- `override` (dict or list[dict]),  containing the sub-modules that not inherit from BaseModule and whose initialization configuration is different from other layers' which are in `'layer'` key. Initializer defined in `type` will work for all layers defined in `layer`, so if sub-modules are not derived Classes of `BaseModule` but can be initialized as same ways of layers in `layer`, it does not need to use `override`. `override` contains:
  - `type` followed by arguments of initializer;
  - `name` to indicate sub-module which will be initialized.

### Initialize parameters

Inherit a new model from `mmcv.runner.BaseModule` or `mmdet.models`  Here we show an example of FooModel.

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

- Initialize model by using `init_cfg` directly in code

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

- Initialize model by using `init_cfg` directly in `mmcv.Sequential` or `mmcv.ModuleList` code

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

- Initialize model by using `init_cfg` in config file

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

### Usage of init_cfg

1. Initialize model by `layer` key

   If we only define `layer`, it just initialize the layer in `layer` key.

   NOTE: Value of `layer` key is the class name with attributes weights and bias of Pytorch, (so such as  `MultiheadAttention layer` is not supported).

- Define `layer` key for initializing module with same configuration.

  ```python
  init_cfg = dict(type='Constant', layer=['Conv1d', 'Conv2d', 'Linear'], val=1)
  # initialize whole module with same configuration
  ```

-  Define `layer` key for initializing layer with different configurations.

  ```python
  init_cfg = [dict(type='Constant', layer='Conv1d', val=1),
              dict(type='Constant', layer='Conv2d', val=2),
              dict(type='Constant', layer='Linear', val=3)]
  # nn.Conv1d will be initialized with dict(type='Constant', val=1)
  # nn.Conv2d will be initialized with dict(type='Constant', val=2)
  # nn.Linear will be initialized with dict(type='Constant', val=3)
  ```

2. Initialize model by `override` key

- When initializing some specific part with its attribute name, we can use `override` key, and the value in `override` will ignore the value in init_cfg.

  ```python
  # layers：
  # self.feat = nn.Conv1d(3, 1, 3)
  # self.reg = nn.Conv2d(3, 3, 3)
  # self.cls = nn.Linear(1,2)

  init_cfg = dict(type='Constant',
                  layer=['Conv1d','Conv2d'], val=1, bias=2,
                  override=dict(type='Constant', name='reg', val=3, bias=4))
  # self.feat and self.cls will be initialized with 	dict(type='Constant', val=1, bias=2)
  # The module called 'reg' will be initialized with dict(type='Constant', val=3, bias=4)
  ```

- If `layer` is None in init_cfg, only sub-module with the name in override will be initialized, and type and other args in override can be omitted.

	```python
	# layers：
	# self.feat = nn.Conv1d(3, 1, 3)
	# self.reg = nn.Conv2d(3, 3, 3)
	# self.cls = nn.Linear(1,2)

	init_cfg = dict(type='Constant', val=1, bias=2, 	override=dict(name='reg'))

	# self.feat and self.cls will be initialized by Pytorch
	# The module called 'reg' will be initialized with dict(type='Constant', val=1, bias=2)
	```

- If we don't define `layer` key or `override` key, it will not initialize anything.
- Invalid usage

	```python
	# It is invalid that override don't have name key
	init_cfg = dict(type='Constant', layer=['Conv1d','Conv2d'], val=1, bias=2,
	            	override=dict(type='Constant', val=3, bias=4))

	# It is also invalid that override has name and other args except type
	init_cfg = dict(type='Constant', layer=['Conv1d','Conv2d'], val=1, bias=2,
	                override=dict(name='reg', val=3, bias=4))
	```

3. Initialize model with the pretrained model

	```python
	init_cfg = dict(type='Pretrained',
                checkpoint='torchvision://resnet50')
	```

More details can refer to the documentation in [MMCV](https://mmcv.readthedocs.io/en/latest/cnn.html#weight-initialization) and MMCV [PR #780](https://github.com/open-mmlab/mmcv/pull/780)
