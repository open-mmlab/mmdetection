# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import contextlib
import logging
import os
import time
from typing import List

import torch

logger = logging.getLogger(__name__)

DEBUG_COMPLETED_TIME = bool(os.environ.get('DEBUG_COMPLETED_TIME', False))


@contextlib.asynccontextmanager
async def completed(trace_name='',
                    name='',
                    sleep_interval=0.05,
                    streams: List[torch.cuda.Stream] = None):
    """Async context manager that waits for work to complete on given CUDA
    streams."""
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
    logger.debug('%s %s starting, streams: %s', trace_name, name, streams)
    grad_enabled_before = torch.is_grad_enabled()
    try:
        yield
    finally:
        current_stream = torch.cuda.current_stream()
        assert current_stream == stream_before_context_switch

        if DEBUG_COMPLETED_TIME:
            cpu_end = time.monotonic()
        for i, stream in enumerate(streams):
            event = end_events[i]
            stream.record_event(event)

        grad_enabled_after = torch.is_grad_enabled()

        # observed change of torch.is_grad_enabled() during concurrent run of
        # async_test_bboxes code
        assert (grad_enabled_before == grad_enabled_after
                ), 'Unexpected is_grad_enabled() value change'

        are_done = [e.query() for e in end_events]
        logger.debug('%s %s completed: %s streams: %s', trace_name, name,
                     are_done, streams)
        with torch.cuda.stream(stream_before_context_switch):
            while not all(are_done):
                await asyncio.sleep(sleep_interval)
                are_done = [e.query() for e in end_events]
                logger.debug(
                    '%s %s completed: %s streams: %s',
                    trace_name,
                    name,
                    are_done,
                    streams,
                )

        current_stream = torch.cuda.current_stream()
        assert current_stream == stream_before_context_switch

        if DEBUG_COMPLETED_TIME:
            cpu_time = (cpu_end - cpu_start) * 1000
            stream_times_ms = ''
            for i, stream in enumerate(streams):
                elapsed_time = start.elapsed_time(end_events[i])
                stream_times_ms += f' {stream} {elapsed_time:.2f} ms'
            logger.info('%s %s %.2f ms %s', trace_name, name, cpu_time,
                        stream_times_ms)


@contextlib.asynccontextmanager
async def concurrent(streamqueue: asyncio.Queue,
                     trace_name='concurrent',
                     name='stream'):
    """Run code concurrently in different streams.

    :param streamqueue: asyncio.Queue instance.

    Queue tasks define the pool of streams used for concurrent execution.
    """
    if not torch.cuda.is_available():
        yield
        return

    initial_stream = torch.cuda.current_stream()

    with torch.cuda.stream(initial_stream):
        stream = await streamqueue.get()
        assert isinstance(stream, torch.cuda.Stream)

        try:
            with torch.cuda.stream(stream):
                logger.debug('%s %s is starting, stream: %s', trace_name, name,
                             stream)
                yield
                current = torch.cuda.current_stream()
                assert current == stream
                logger.debug('%s %s has finished, stream: %s', trace_name,
                             name, stream)
        finally:
            streamqueue.task_done()
            streamqueue.put_nowait(stream)
