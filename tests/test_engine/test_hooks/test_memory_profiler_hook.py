# Copyright (c) OpenMMLab. All rights reserved.
from collections import namedtuple
from unittest import TestCase
from unittest.mock import Mock, patch


class TestMemoryProfilerHook(TestCase):

    def test_memory_profiler_hook(self):
        from mmdet.engine.hooks import MemoryProfilerHook

        def _mock_virtual_memory():
            virtual_memory_type = namedtuple(
                'virtual_memory', ['total', 'available', 'percent', 'used'])
            return virtual_memory_type(
                total=270109085696,
                available=250416816128,
                percent=7.3,
                used=17840881664)

        def _mock_swap_memory():
            swap_memory_type = namedtuple('swap_memory', [
                'total',
                'used',
                'percent',
            ])
            return swap_memory_type(total=8589930496, used=0, percent=0.0)

        def _mock_memory_usage():
            return [40.22265625]

        mock_virtual_memory = Mock(return_value=_mock_virtual_memory())
        mock_swap_memory = Mock(return_value=_mock_swap_memory())
        mock_memory_usage = Mock(return_value=_mock_memory_usage())

        @patch('psutil.swap_memory', mock_swap_memory)
        @patch('psutil.virtual_memory', mock_virtual_memory)
        @patch('memory_profiler.memory_usage', mock_memory_usage)
        def _test_after_iter():
            hook = MemoryProfilerHook(2)
            runner = Mock()

            assert not mock_memory_usage.called
            assert not mock_swap_memory.called
            assert not mock_memory_usage.called
            hook._after_iter(runner, 0)
            assert not mock_memory_usage.called
            assert not mock_swap_memory.called
            assert not mock_memory_usage.called
            hook._after_iter(runner, 1)
            assert mock_memory_usage.called
            assert mock_swap_memory.called
            assert mock_memory_usage.called

        _test_after_iter()
