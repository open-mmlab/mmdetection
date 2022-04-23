# Tutorial 13: Useful Hooks

what is hook? what is the function of hook? How to use?

usage
```python
custom_hooks = [
    dict(type='MemoryProfilerHook', interval=50),
    dict(type='CheckInvalidLossHook', interval=50)
]
```

## Memory profiler hook
Memory profiler hook records memory information including virtual memory, swap memory, and the memory of the current process. This hook can help us grasp the memory usage of the system and discover potential memory leak bugs. To use this hook, we must install `memory_profiler`, `psutil` by `pip install memory_profiler psutil`. The following is the result of `MemoryProfilerHook`.

```text
2022-04-21 08:49:56,881 - mmdet - INFO - Memory information available_memory: 246360 MB, used_memory: 9407 MB, memory_utilization: 4.4 %, available_swap_memory: 5740MB, used_swap_memory: 2452 MB, swap_memory_utilization: 29.9 %, current_process_memory: 5434 MB
```
