import contextlib
import sys
import time

import torch

if sys.version_info >= (3, 7):

    @contextlib.contextmanager
    def profile_time(trace_name,
                     name,
                     enabled=True,
                     stream=None,
                     end_stream=None):
        """Print time spent by CPU and GPU.

        Useful as a temporary context manager to find sweet spots of
        code suitable for async implementation.

        """
        if (not enabled) or not torch.cuda.is_available():
            yield
            return
        stream = stream if stream else torch.cuda.current_stream()
        end_stream = end_stream if end_stream else stream
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream.record_event(start)
        try:
            cpu_start = time.monotonic()
            yield
        finally:
            cpu_end = time.monotonic()
            end_stream.record_event(end)
            end.synchronize()
            cpu_time = (cpu_end - cpu_start) * 1000
            gpu_time = start.elapsed_time(end)
            msg = f'{trace_name} {name} cpu_time {cpu_time:.2f} ms '
            msg += f'gpu_time {gpu_time:.2f} ms stream {stream}'
            print(msg, end_stream)
