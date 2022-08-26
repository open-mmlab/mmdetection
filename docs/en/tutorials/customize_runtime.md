# Tutorial 5: Customize Runtime Settings

## Customize optimization settings

Optimization related configuration is now all managed by `optim_wrapper` which usually has three fields: `optimizer`, `paramwise_cfg`, `clip_grad`. See the example below, where `Adamw` is used as an `optimizer`, the learning rate of the backbone is reduced by a factor of 10, and gradient clipping is added.

```python
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),

    # Parameter-level learning rate and weight decay settings
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0),

    # gradient clipping
    clip_grad=dict(max_norm=0.01, norm_type=2))
```

### Customize optimizer supported by Pytorch

We already support to use all the optimizers implemented by PyTorch, and the only modification is to change the `optimizer` field in `optim_wrapper` field of config files. For example, if you want to use `ADAM` (note that the performance could drop a lot), the modification could be as the following.

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.0003, weight_decay=0.0001))
```

To modify the learning rate of the model, the users only need to modify the `lr` in `optimizer`. The users can directly set arguments following the [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) of PyTorch.

### Customize self-implemented optimizer

#### 1. Define a new optimizer

A customized optimizer could be defined as following.

Assume you want to add a optimizer named `MyOptimizer`, which has arguments `a`, `b`, and `c`.
You need to create a new directory named `mmdet/engine/optimizers`. And then implement the new optimizer in a file, e.g., in `mmdet/engine/optimizers/my_optimizer.py`:

```python
from mmdet.registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)

```

#### 2. Add the optimizer to registry

To find the above module defined above, this module should be imported into the main namespace at first. There are two options to achieve it.

- Modify `mmdet/engine/optimizers/__init__.py` to import it.

  The newly defined module should be imported in `mmdet/engine/optimizers/__init__.py` so that the registry will find the new module and add it:

```python
from .my_optimizer import MyOptimizer
```

- Use `custom_imports` in the config to manually import it

```python
custom_imports = dict(imports=['mmdet.engine.optimizers.my_optimizer'], allow_failed_imports=False)
```

The module `mmdet.engine.optimizers.my_optimizer` will be imported at the beginning of the program and the class `MyOptimizer` is then automatically registered.
Note that only the package containing the class `MyOptimizer` should be imported.
`mmdet.engine.optimizers.my_optimizer.MyOptimizer` **cannot** be imported directly.

Actually users can use a totally different file directory structure using this importing method, as long as the module root can be located in `PYTHONPATH`.

#### 3. Specify the optimizer in the config file

Then you can use `MyOptimizer` in `optimizer` field in `optim_wrapper` field of config files. In the configs, the optimizers are defined by the field `optimizer` like the following:

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
```

To use your own optimizer, the field can be changed to

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value))
```

### Customize optimizer constructor

Some models may have some parameter-specific settings for optimization, e.g. weight decay for BatchNorm layers.
The users can do those fine-grained parameter tuning through customizing optimizer constructor.

```python
from mmengine.optim import DefaultOptiWrapperConstructor

from mmdet.registry import OPTIM_WRAPPER_CONSTRUCTORS
from .my_optimizer import MyOptimizer


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MyOptimizerConstructor(DefaultOptimWrapperConstructor):

    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None):

    def __call__(self, model: nn.Module) -> OptimWrapper:

        return optim_wrapper

```

The default optimizer constructor is implemented [here](https://github.com/open-mmlab/mmengine/blob/1a8f0139373ab3d5e0a7012f42e3b345776c0b72/mmengine/optim/optimizer/default_constructor.py#L18), which could also serve as a template for new optimizer constructor.

### Additional settings

Tricks not implemented by the optimizer should be implemented through optimizer constructor (e.g., set parameter-wise learning rates) or hooks. We list some common settings that could stabilize the training or accelerate the training. Feel free to create PR, issue for more settings.

- __Use gradient clip to stabilize training__:
  Some models need gradient clip to clip the gradients to stabilize the training process. An example is as below:

  ```python
  optim_wrapper = dict(
      _delete_=True, grad_clip=dict(max_norm=35, norm_type=2)
  ```

  If your config inherits the base config which already sets the `optim_wrapper`, you might need `_delete_=True` to override the unnecessary settings. See the [config documentation](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html) for more details.

- __Use momentum schedule to accelerate model convergence__:
  We support momentum scheduler to modify model's momentum according to learning rate, which could make the model converge in a faster way.
  Momentum scheduler is usually used with LR scheduler, for example, the following config is used in 3D detection to accelerate convergence.
  For more details, please refer to the implementation of [CosineAnnealingLR](https://github.com/open-mmlab/mmengine/blob/6ebb7ed481614b8ef08aabe27f6d88f750eb65dd/mmengine/optim/scheduler/lr_scheduler.py#L42) and [CosineAnnealingBetas](https://github.com/open-mmlab/mmdetection3d/blob/4b4a7e4d671c2d86cf8262f7f8b664633e14ce73/mmdet3d/engine/scheduler/betas_scheduler.py#L23).

  ```python
  param_scheduler = [
      # learning rate scheduler
      dict(
          type='CosineAnnealingLR',
          T_max=8,
          eta_min=lr * 10,
          begin=0,
          end=8,
          by_epoch=True,
          convert_to_iter_based=True),
      dict(
          type='CosineAnnealingLR',
          T_max=12,
          eta_min=lr * 1e-4,
          begin=8,
          end=20,
          by_epoch=True,
          convert_to_iter_based=True),
      # momentum scheduler
      dict(
          type='CosineAnnealingBetas',
          T_max=8,
          eta_min=0.85 / 0.95,
          begin=0,
          end=8,
          by_epoch=True,
          convert_to_iter_based=True),
      dict(
          type='CosineAnnealingBetas',
          T_max=12,
          eta_min=1,
          begin=8,
          end=20,
          by_epoch=True,
          convert_to_iter_based=True)
  ]
  ```

## Customize training schedules

By default we use step learning rate with 1x schedule, this calls [MultiStepLR](https://github.com/open-mmlab/mmengine/blob/1a8f0139373ab3d5e0a7012f42e3b345776c0b72/mmengine/optim/scheduler/lr_scheduler.py#L139) in MMEngine.
We support many other learning rate schedule [here](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py), such as `CosineAnnealingLR` and `PolyLR` schedule. Here are some examples

- Poly schedule:

  ```python
  param_scheduler = [
      dict(
          type='PolyLR',
          power=0.9,
          eta_min=1e-4,
          begin=0,
          end=8,
          by_epoch=True)]
  ```

- ConsineAnnealing schedule:

  ```python
  param_scheduler = [
      dict(
          type='CosineAnnealingLR',
          T_max=8,
          eta_min=lr * 1e-5,
          begin=0,
          end=8,
          by_epoch=True)]

  ```

## Customize workflow

# TODO

```python
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

## Customize hooks

### Customize self-implemented hooks

#### 1. Implement a new hook

There are some occasions when the users might need to implement a new hook. MMDetection supports customized hooks in training (#3395) since v2.3.0. Thus the users could implement a hook directly in mmdet or their mmdet-based codebases and use the hook by only modifying the config in training.
Before v2.3.0, the users need to modify the code to get the hook registered before training starts.
Here we give an example of creating a new hook in mmdet and using it in training.

```python
from mmengne.hooks import HOOKS, Hook


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):

    def before_run(self, runner) -> None:

    def after_run(self, runner) -> None:

    def before_train(self, runner) -> None:

    def after_train(self, runner) -> None:

    def before_val(self, runner) -> None:

    def after_val(self, runner) -> None:

    def before_test(self, runner) -> None:

    def after_test(self, runner) -> None:

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:

    def before_train_epoch(self, runner) -> None:

    def before_val_epoch(self, runner) -> None:

    def before_test_epoch(self, runner) -> None:

    def after_train_epoch(self, runner) -> None:

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:

    def before_val_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None) -> None:

    def before_test_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None) -> None:

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence[BaseDataElement]] = None) \
            -> None:

    def after_test_iter(
            self,
            runner,
            batch_idx: int,
            data_batch: DATA_BATCH = None,
            outputs: Optional[Sequence[BaseDataElement]] = None) -> None:

```

Depending on the functionality of the hook, the users need to specify what the hook will do at each stage of the training in `before_run`, `after_run`, `before_epoch`, `after_epoch`, `before_iter`, and etc..

#### 2. Register the new hook

Then we need to make `MyHook` imported. Assuming the file is in `mmdet/engine/hooks/my_hook.py` there are two ways to do that:

- Modify `mmdet/engine/hooks/__init__.py` to import it.

  The newly defined module should be imported in `mmdet/engine/hooks/__init__.py` so that the registry will find the new module and add it:

```python
from .my_hook import MyHook
```

- Use `custom_imports` in the config to manually import it

```python
custom_imports = dict(imports=['mmdet.engine.hooks.my_hook'], allow_failed_imports=False)
```

#### 3. Modify the config

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

You can also set the priority of the hook by adding key `priority` to `'NORMAL'` or `'HIGHEST'` as below

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

By default the hook's priority is set as `NORMAL` during registration.

### Use hooks implemented in MMEngine

If the hook is already implemented in MMEngine, you can directly modify the config to use the hook as below

#### 4. Example: `NumClassCheckHook`

We implement a customized hook named  [NumClassCheckHook](https://github.com/open-mmlab/mmdetection/blob/test-3.0.0rc0/mmdet/engine/hooks/num_class_check_hook.py) to check whether the `num_classes` in head matches the length of `CLASSES` in `dataset`.

We set it in [default_runtime.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/default_runtime.py).

```python
custom_hooks = [dict(type='NumClassCheckHook')]
```

### Modify default runtime hooks

There are some common hooks that are registered through `default_hooks`, they are

- IterTimerHook
- LoggerHook
- ParamSchedulerHook
- CheckpointHook
- DistSamplerSeedHook
- DetVisualizationHook

In those hooks, only the logger hook has the `VERY_LOW` priority, others' priority are `NORMAL`.
The above-mentioned tutorials already covers how to modify `optimizer_config`, `momentum_config`, and `lr_config`.
Here we reveals how what we can do with `LoggerHook` and `CheckpointHook`.

#### Checkpoint config

The MMengine runner will use `checkpoint_config` to initialize [`CheckpointHook`](https://github.com/open-mmlab/mmengine/blob/6ebb7ed481614b8ef08aabe27f6d88f750eb65dd/mmengine/hooks/checkpoint_hook.py#L19).

```python
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1))
```

The users could set `max_keep_ckpts` to only save only small number of checkpoints or decide whether to store state dict of optimizer by `save_optimizer`. More details of the arguments are [here](https://github.com/open-mmlab/mmengine/blob/6ebb7ed481614b8ef08aabe27f6d88f750eb65dd/mmengine/hooks/checkpoint_hook.py#L19)

#### Log config

The `log_config` wraps multiple logger hooks and enables to set intervals. Now MMCV supports `WandbLoggerHook`, `MlflowLoggerHook`, and `TensorboardLoggerHook`.
The detail usages can be found in the [doc](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook).

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

# TODO Visualizer
