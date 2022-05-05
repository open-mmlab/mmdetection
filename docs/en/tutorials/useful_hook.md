# Tutorial 13: Useful Hooks

MMDetection and MMCV provide users with various useful hooks including log hooks, evaluation hooks, NumClassCheckHook and etc. . This tutorial mainly introduce function and usage of hooks in MMDetection\. For how to add custom hook to MMDetection, please read [customize_runtime](https://mmdetection.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-self-implemented-hooks),  and for hooks in MMCV, please read [docs in MMCV](https://github.com/open-mmlab/mmcv/blob/master/docs/en/understand_mmcv/runner.md).

## CheckInvalidLossHook

## EvalHook and DistEvalHook

## ExpMomentumEMAHook and LinearMomentumEMAHook

## NumClassCheckHook

## [MemoryProfilerHook](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/hook/memory_profiler_hook.py)

Memory profiler hook records memory information including virtual memory, swap memory, and the memory of the current process. This hook can help us grasp the memory usage of the system and discover potential memory leak bugs. To use this hook, we must install `memory_profiler` and `psutil` by `pip install memory_profiler psutil` first.

### Usage

To use this hook, we should add the following code to config file.

```python
custom_hooks = [
    dict(type='MemoryProfilerHook', interval=50)
]
```

### Result

The following is the result of `MemoryProfilerHook`.

```text
2022-04-21 08:49:56,881 - mmdet - INFO - Memory information available_memory: 246360 MB, used_memory: 9407 MB, memory_utilization: 4.4 %, available_swap_memory: 5740 MB, used_swap_memory: 2452 MB, swap_memory_utilization: 29.9 %, current_process_memory: 5434 MB
```

## SetEpochInfoHook

## SyncNormHook

## SyncRandomSizeHook

## YOLOXLrUpdaterHook

## YOLOXModeSwitchHook
