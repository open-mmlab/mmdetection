# coding: utf-8
import asyncio
import contextlib
import logging
import os
import time
from typing import List

import torch

logger = logging.getLogger(__name__)

DEBUG_COMPLETED_TIME = bool(os.environ.get('DEBUG_COMPLETED_TIME', False))
DEBUG_COMPLETED = bool(os.environ.get('DEBUG_COMPLETED', False))


@contextlib.asynccontextmanager
async def completed(trace_name="",
                    name="",
                    sleep_interval=0.05,
                    streams: List[torch.cuda.Stream] = None):
    """
    Async context manager that waits for work to complete on
    given CUDA streams.

    """
    if not torch.cuda.is_available():
        yield
        return

    stream_before_context_switch = torch.cuda.current_stream()
    if not streams:
        streams = [stream_before_context_switch]
    else:
        streams = [s if s else stream_before_context_switch for s in streams]

    end_events = [
        torch.cuda.Event(enable_timing=DEBUG_COMPLETED_TIME) for _ in streams
    ]

    if DEBUG_COMPLETED_TIME:
        start = torch.cuda.Event(enable_timing=True)
        stream_before_context_switch.record_event(start)

        cpu_start = time.monotonic()
    if DEBUG_COMPLETED:
        logger.info("%s %s starting, streams: %s", trace_name, name, streams)
    grad_enabled_before = torch.is_grad_enabled()
    try:
        yield
    finally:
        if DEBUG_COMPLETED_TIME:
            cpu_end = time.monotonic()
        for i, stream in enumerate(streams):
            event = end_events[i]
            stream.record_event(event)

        grad_enabled_after = torch.is_grad_enabled()

        # observed change of torch.is_grad_enabled() during concurrent run of
        # async_test_bboxes code
        assert grad_enabled_before == grad_enabled_after, \
            "Unexpected is_grad_enabled() value change"

        device_before_context_switch = torch.cuda.current_device()

        are_done = [e.query() for e in end_events]
        if DEBUG_COMPLETED:
            logger.info("%s %s completed: %s streams: %s", trace_name, name,
                        are_done, streams)
        while not all(are_done):
            await asyncio.sleep(sleep_interval)
            are_done = [e.query() for e in end_events]
            if DEBUG_COMPLETED:
                logger.info("%s %s completed: %s streams: %s", trace_name,
                            name, are_done, streams)

        current_device = torch.cuda.current_device()
        current_stream = torch.cuda.current_stream()

        if current_device != device_before_context_switch:
            torch.cuda.set_device(device_before_context_switch)
        if current_stream != stream_before_context_switch:
            torch._C._cuda_setStream(stream_before_context_switch._cdata)

        if DEBUG_COMPLETED_TIME:
            cpu_time = (cpu_end - cpu_start) * 1000
            stream_times_ms = ""
            for i, stream in enumerate(streams):
                elapsed_time = start.elapsed_time(end_events[i])
                stream_times_ms += f" {stream} {elapsed_time:.2f} ms"
            logger.info(f"{trace_name} {name} cpu_time {cpu_time:.2f} ms" +
                        stream_times_ms)


@contextlib.asynccontextmanager
async def completed(trace_name="",
                    name="",
