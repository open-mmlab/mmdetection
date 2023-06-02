# 实用的钩子

MMDetection 和 MMEngine 为用户提供了多种多样实用的钩子（Hook），包括 `MemoryProfilerHook`、`NumClassCheckHook` 等等。
这篇教程介绍了 MMDetection 中实现的钩子功能及使用方式。若使用 MMEngine 定义的钩子请参考 [MMEngine 的钩子API文档](https://github.com/open-mmlab/mmengine/tree/main/docs/en/tutorials/hook.md).

## CheckInvalidLossHook

## NumClassCheckHook

## MemoryProfilerHook

[内存分析钩子](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/engine/hooks/memory_profiler_hook.py)
记录了包括虚拟内存、交换内存、当前进程在内的所有内存信息，它能够帮助捕捉系统的使用状况与发现隐藏的内存泄露问题。为了使用这个钩子，你需要先通过 `pip install memory_profiler psutil` 命令安装 `memory_profiler` 和 `psutil`。

### 使用

为了使用这个钩子，使用者需要添加如下代码至 config 文件

```python
custom_hooks = [
    dict(type='MemoryProfilerHook', interval=50)
]
```

### 结果

在训练中，你会看到 `MemoryProfilerHook` 记录的如下信息：

```text
The system has 250 GB (246360 MB + 9407 MB) of memory and 8 GB (5740 MB + 2452 MB) of swap memory in total. Currently 9407 MB (4.4%) of memory and 5740 MB (29.9%) of swap memory were consumed. And the current training process consumed 5434 MB of memory.
```

```text
2022-04-21 08:49:56,881 - mmengine - INFO - Memory information available_memory: 246360 MB, used_memory: 9407 MB, memory_utilization: 4.4 %, available_swap_memory: 5740 MB, used_swap_memory: 2452 MB, swap_memory_utilization: 29.9 %, current_process_memory: 5434 MB
```

## SetEpochInfoHook

## SyncNormHook

## SyncRandomSizeHook

## YOLOXLrUpdaterHook

## YOLOXModeSwitchHook

## 如何实现自定义钩子

通常，从模型训练的开始到结束，共有20个点位可以执行钩子。我们可以实现自定义钩子在不同点位执行，以便在训练中实现自定义操作。

- global points: `before_run`, `after_run`
- points in training: `before_train`, `before_train_epoch`, `before_train_iter`, `after_train_iter`, `after_train_epoch`, `after_train`
- points in validation: `before_val`, `before_val_epoch`, `before_val_iter`, `after_val_iter`, `after_val_epoch`, `after_val`
- points at testing: `before_test`, `before_test_epoch`, `before_test_iter`, `after_test_iter`, `after_test_epoch`,  `after_test`
- other points: `before_save_checkpoint`, `after_save_checkpoint`

比如，我们要实现一个检查 loss 的钩子，当损失为 NaN 时自动结束训练。我们可以把这个过程分为三步：

1. 在 MMEngine 实现一个继承于 `Hook` 类的新钩子，并实现 `after_train_iter` 方法用于检查每 `n` 次训练迭代后损失是否变为 NaN 。
2. 使用 `@HOOKS.register_module()` 注册实现好了的自定义钩子，如下列代码所示。
3. 在配置文件中添加 `custom_hooks = [dict(type='MemoryProfilerHook', interval=50)]`

```python
from typing import Optional

import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class CheckInvalidLossHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval: int = 50) -> None:
        self.interval = interval

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        """Regularly check whether the loss is valid every n iterations.

        Args:
            runner (:obj:`Runner`): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict, Optional): Data from dataloader.
                Defaults to None.
            outputs (dict, Optional): Outputs from model. Defaults to None.
        """
        if self.every_n_train_iters(runner, self.interval):
            assert torch.isfinite(outputs['loss']), \
                runner.logger.info('loss become infinite or NaN!')
```

请参考 [自定义训练配置](../advanced_guides/customize_runtime.md) 了解更多与自定义钩子相关的内容。
