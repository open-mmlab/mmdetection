# Tutorial 13: Useful Hooks

MMDetection and MMCV provide users with various useful hooks including log hooks, evaluation hooks, NumClassCheckHook, etc. This tutorial introduces the functionalities and usages of hooks implemented in MMDetection. For using hooks in MMCV, please read the [API documentation in MMCV](https://github.com/open-mmlab/mmcv/blob/master/docs/en/understand_mmcv/runner.md).

## CheckInvalidLossHook

## EvalHook and DistEvalHook

## ExpMomentumEMAHook and LinearMomentumEMAHook

## NumClassCheckHook

## [MemoryProfilerHook](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/hook/memory_profiler_hook.py)

Memory profiler hook records memory information including virtual memory, swap memory, and the memory of the current process. This hook helps grasp the memory usage of the system and discover potential memory leak bugs. To use this hook, users should install `memory_profiler` and `psutil` by `pip install memory_profiler psutil` first.

### Usage

To use this hook, users should add the following code to the config file.

```python
custom_hooks = [
    dict(type='MemoryProfilerHook', interval=50)
]
```

### Result

During training, you can see the messages in the log recorded by `MemoryProfilerHook` as below. The system has 250 GB (246360 MB + 9407 MB) of memory and 8 GB (5740 MB + 2452 MB) of swap memory in total. Currently 9407 MB (4.4%) of memory and 5740 MB (29.9%) of swap memory were consumed. And the current training process consumed 5434 MB of memory.

```text
2022-04-21 08:49:56,881 - mmdet - INFO - Memory information available_memory: 246360 MB, used_memory: 9407 MB, memory_utilization: 4.4 %, available_swap_memory: 5740 MB, used_swap_memory: 2452 MB, swap_memory_utilization: 29.9 %, current_process_memory: 5434 MB
```

## SetEpochInfoHook

## SyncNormHook

## SyncRandomSizeHook

## YOLOXLrUpdaterHook

## YOLOXModeSwitchHook

## How to implement a custom hook

In general, there are 10 points where hooks can be inserted from the beginning to the end of model training. The users can implement custom hooks and insert them at different points in the process of training to do what they want.

- global points: `before_run`, `after_run`
- points in training: `before_train_epoch`, `before_train_iter`, `after_train_iter`, `after_train_epoch`
- points in validation: `before_val_epoch`, `before_val_iter`, `after_val_iter`, `after_val_epoch`

For example, users can implement a hook to check loss and terminate training when loss goes NaN. To achieve that, there are three steps to go:

1. Implement a new hook that inherits the `Hook` class in MMCV, and implement `after_train_iter` method which checks whether loss goes NaN after every `n` training iterations.
2. The implemented hook should be registered in `HOOKS` by `@HOOKS.register_module()` as shown in the code below.
3. Add `custom_hooks = [dict(type='CheckInvalidLossHook', interval=50)]` in the config file.

```python
import torch
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class CheckInvalidLossHook(Hook):
    """Check invalid loss hook.
    This hook will regularly check whether the loss is valid
    during training.
    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval=50):
        self.interval = interval

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            assert torch.isfinite(runner.outputs['loss']), \
                runner.logger.info('loss become infinite or NaN!')
```

Please read [customize_runtime](https://mmdetection.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-self-implemented-hooks) for more about implementing a custom hook.
