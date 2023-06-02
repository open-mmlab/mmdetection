# 自定义训练配置

## 自定义优化相关的配置

优化相关的配置现在已全部集成到 `optim_wrapper` 中，通常包含三个域：`optimizer`, `paramwise_cfg`，`clip_grad`，具体细节见 [OptimWrapper](https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.md)。下面这个例子中，使用了 `AdamW` 作为优化器，主干部分的学习率缩小到原来的十分之一，以及添加了梯度裁剪。

```python
optim_wrapper = dict(
    type='OptimWrapper',
    # 优化器
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),

    # 参数层面的学习率和正则化设置
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0),

    # 梯度裁剪
    clip_grad=dict(max_norm=0.01, norm_type=2))
```

### 自定义 Pytorch 中优化器设置

我们已经支持了 Pytorch 中实现的所有优化器，要使用这些优化器唯一要做就是修改配置文件中的 `optimi_wrapper` 中的 `optimzer` 域。比如，如果想要使用 `ADAM` 作为优化器（可能会导致性能下降），所需要做的修改如下。

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.0003, weight_decay=0.0001))
```

要修改模型的学习率，用户只需要修改 `optimizer` 中的 `lr` 域。用户可以直接参考 PyToch 的 [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) 来进行参数的设置。

### 自定义优化器

#### 1. 定义一个新优化器

自定义优化器可以定义的方式如下：

假设你想要添加一个名为 `MyOptimizer` 的优化器，它包含三个参数 `a`，`b`，`c`。你需要新建一个名为
`mmdet/engine/optimizers` 的文件夹。然后在文件（比如，`mmdet/engine/optimizers/my_optimizer.py`）实现一个新的优化器。

```python
from mmdet.registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)

```

#### 2. 导入自定义的优化器

为了能找到上面的所定义的模块，这个模块必须要先导入到主命名空间中。有两种方式可以实现这一点。

- 修改 `mmdet/engine/optimizers/__init__.py` 来导入模块。

  新定义的模块必须导入到 `mmdet/engine/optimizers/__init__.py`，这样注册器才能找到该模块并添加它。

```python
from .my_optimizer import MyOptimizer
```

- 在配置文件使用 `custom_imports` 来手动导入模块。

```python
custom_imports = dict(imports=['mmdet.engine.optimizers.my_optimizer'], allow_failed_imports=False)
```

`mmdet.engine.optimizers.my_optimizer` 模块将在程序开始时导入，之后 `MyOptimizer` 类会被自动注册。注意：应该导入 `MyOptimizer` 所在的文件，即 `mmdet.engine.optimizers.my_optimizer`，而不是 `mmdet.engine.optimizers.my_optimizer.MyOptimizer`。

实际上，用户也可以在别的目录结构下来进行导入模块，只要改模块可以在 `PYTHONPATH` 中找到。

#### 3. 在配置文件中指定优化器

接下来，你可以在配置文件中的 `optim_wrapper` 域中的中 `optimizer` 域中设置你实现的优化器 `MyOptimizer`。在配置文件中，优化器在 `optimizer` 域中的配置方式如下：

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
```

为了使用你的优化器，可以进行如下修改

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value))
```

### 自定义优化器包装构造类

一些模型可能存在一些特定参数的优化设置，比如，BN 层的权重衰减。用户可以通过自定义优化器包装构造类来实现这些精细化的参数调整。

```python
from mmengine.optim import DefaultOptiWrapperConstructor

from mmdet.registry import OPTIM_WRAPPER_CONSTRUCTORS
from .my_optimizer import MyOptimizer


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MyOptimizerWrapperConstructor(DefaultOptimWrapperConstructor):

    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None):

    def __call__(self, model: nn.Module) -> OptimWrapper:

        return optim_wrapper

```

优化器包装构造类的具体实现见[这里](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py#L18)，用户以它为模板，来实现新的优化器包装构造类。

### 额外的设置

一些没有被优化器实现的技巧（比如，参数层面的学习率设置）应该通过优化器包装构造类来实现或者钩子。我们列出了一些常用的设置用于稳定训练或者加速训练。请随意创建 PR，发布更多设置。

- __使用梯度裁剪来稳定训练__:
  一些模型需要进行梯度裁剪来稳定训练过程，例子如下：

  ```python
  optim_wrapper = dict(
      _delete_=True, clip_grad=dict(max_norm=35, norm_type=2))
  ```

  如果你的配置已经集成了基础配置（包含了 `optim_wrapper` 的配置），那么你需要添加 `_delete_=True` 来覆盖掉不需要的设置。具体见[配置相关的文档](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html)。

- __使用动量调度加速模型收敛__:
  我们支持动量调度器根据学习率修改模型的动量，这可以使模型以更快的方式收敛。动量调度器通常与学习率调度器一起使用，例如 [3D 检测](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/schedules/cyclic-20e.py) 中使用以下配置以加速收敛。
  更多细节请参考 [CosineAnnealingLR](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L43) 和 [CosineAnnealingMomentum](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/momentum_scheduler.py#L71) 的具体实现。

  ```python
  param_scheduler = [
      # 学习率调度器
      # 在前 8 个 epoch, 学习率从 0 增大到 lr * 10
      # 在接下来 12 个 epoch, 学习率从 lr * 10 减小到 lr * 1e-4
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
      # 动量调度器
      # 在前 8 个 epoch, 动量从 0 增大到 0.85 / 0.95
      # 在接下来 12 个 epoch, 学习率从 0.85 / 0.95 增大到 1
      dict(
          type='CosineAnnealingMomentum',
          T_max=8,
          eta_min=0.85 / 0.95,
          begin=0,
          end=8,
          by_epoch=True,
          convert_to_iter_based=True),
      dict(
          type='CosineAnnealingMomentum',
          T_max=12,
          eta_min=1,
          begin=8,
          end=20,
          by_epoch=True,
          convert_to_iter_based=True)
  ]
  ```

## 自定义训练策略

默认情况下，我们使用 1x 的学习率调整策略，这会条用 MMEngine 中的 [MultiStepLR](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L139)。
我们支持许多其他学习率调整策略，具体见[这里](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py)，例如 `CosineAnnealingLR` 和 `PolyLR` 策略。下面有些例子

- 多项式学习率调整策略:

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

- 余弦退火学习率调整策略

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

## 自定义训练循环

默认情况下，在 `train_cfg` 中使用 `EpochBasedTrainLoop`，并且在每个 epoch 训练之后进行验证，如下所示。

```python
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_begin=1, val_interval=1)
```

实际上，[`IterBasedTrainLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L183%5D) 和\[`EpochBasedTrainLoop`\](https:// github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L18) 支持动态区间的方式进行验证，见下例。

```python
# 在第 365001 次迭代之前，我们每 5000 次迭代进行一次评估。
# 在第 365000 次迭代后，我们每 368750 次迭代进行一次评估，
# 这意味着我们在训练结束时进行评估。

interval = 5000
max_iters = 368750
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
```

## 自定义钩子

### 自定义自行实现的钩子

#### 1. 实现一个新的钩子

MMEngine 提供了许多有用的[钩子](https://mmdetection.readthedocs.io/en/latest/tutorials/hooks.html)，但在某些情况下用户可能需要实现新的钩子。MMDetection 在 v3.0 中支持自定义钩子。因此，用户可以直接在 mmdet 或其基于 mmdet 的代码库中实现钩子，并通过仅在训练中修改配置来使用钩子。
这里我们给出一个在 mmdet 中创建一个新的钩子并在训练中使用它的例子。

```python
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):

    def before_run(self, runner) -> None:

    def after_run(self, runner) -> None:

    def before_train(self, runner) -> None:

    def after_train(self, runner) -> None:

    def before_train_epoch(self, runner) -> None:

    def after_train_epoch(self, runner) -> None:

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
```

根据钩子的功能，用户需要在 `before_run`、`after_run`、`before_train`、`after_train`、`before_train_epoch`、`after_train_epoch`、`before_train_iter` 和 `after_train_iter`。还有更多可以插入钩子的点，更多细节请参考 [base hook class](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/hook.py#L9)。

#### 2. 注册新钩子

然后我们需要导入 `MyHook`。假设该文件位于 `mmdet/engine/hooks/my_hook.py` 中，有两种方法可以做到这一点：

- 修改 `mmdet/engine/hooks/__init__.py` 以导入它。

  新定义的模块应该在 `mmdet/engine/hooks/__init__.py` 中导入，以便注册表找到新模块并添加它：

```python
from .my_hook import MyHook
```

- 在配置中使用 `custom_imports` 手动导入它

```python
custom_imports = dict(imports=['mmdet.engine.hooks.my_hook'], allow_failed_imports=False)
```

#### 3. 修改配置

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

你还可以通过修改键 `priority` 的值为 `NORMAL` 或 `HIGHEST` 来设置挂钩的优先级，如下所示

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

默认情况下，钩子的优先级在注册期间设置为 `NORMAL`。

### 使用 MMDetection 中实现的钩子

如果 MMDetection 中已经实现了该钩子，你可以直接修改配置以使用该钩子，如下所示

#### 例子: `NumClassCheckHook`

我们实现了一个名为 [NumClassCheckHook](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/engine/hooks/num_class_check_hook.py) 的自定义钩子来检查 `num_classes` 是否在 head 中和 `dataset` 中的 `classes` 的长度相匹配。

我们在 [default_runtime.py](https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/default_runtime.py) 中设置它。

```python
custom_hooks = [dict(type='NumClassCheckHook')]
```

### 修改默认运行时钩子

有一些常见的钩子是通过 `default_hooks` 注册的，它们是

- `IterTimerHook`：记录 “data_time” 用于加载数据和 “time” 用于模型训练步骤的钩子。
- `LoggerHook`：从`Runner`的不同组件收集日志并将它们写入终端、JSON文件、tensorboard和 wandb 等的钩子。
- `ParamSchedulerHook`：更新优化器中一些超参数的钩子，例如学习率和动量。
- `CheckpointHook`：定期保存检查点的钩子。
- `DistSamplerSeedHook`：为采样器和批处理采样器设置种子的钩子。
- `DetVisualizationHook`：用于可视化验证和测试过程预测结果的钩子。

`IterTimerHook`、`ParamSchedulerHook` 和 `DistSamplerSeedHook` 很简单，通常不需要修改，所以这里我们将展示如何使用 `LoggerHook`、`CheckpointHook` 和 `DetVisualizationHook`。

#### CheckpointHook

除了定期保存检查点，[`CheckpointHook`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L19) 提供了其他选项，例如`max_keep_ckpts`、`save_optimizer ` 等。用户可以设置 `max_keep_ckpts` 只保存少量检查点或通过 `save_optimizer` 决定是否存储优化器的状态字典。参数的更多细节在[这里](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L19)可以找到。

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_optimizer=True))
```

#### LoggerHook

`LoggerHook` 可以设置间隔。详细用法可以在 [docstring](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py#L18) 中找到。

```python
default_hooks = dict(logger=dict(type='LoggerHook', interval=50))
```

#### DetVisualizationHook

`DetVisualizationHook` 使用 `DetLocalVisualizer` 来可视化预测结果，`DetLocalVisualizer` 支持不同的后端，例如 `TensorboardVisBackend` 和 `WandbVisBackend` （见 [docstring](https://github.com/open-mmlab/mmengine/blob/main/mmengine/visualization/vis_backend.py) 了解更多细节）。用户可以添加多个后端来进行可视化，如下所示。

```python
default_hooks = dict(
    visualization=dict(type='DetVisualizationHook', draw=True))

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```
