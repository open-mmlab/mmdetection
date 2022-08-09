# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import torch
from mmengine.runner import BaseLoop
from torch.utils.data import DataLoader

from mmdet.registry import LOOPS


@LOOPS.register_module()
class MultiStageTrainLoop(BaseLoop):
    """A wrapper to execute a sequence of train loops.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Sequence[Union[DataLoader, dict]]): A dataloader object
            or a dict to build a dataloader.
        stages (Sequence[dict]): A list of train loop configs for each stage.
    """

    def __init__(self, runner, dataloader: Sequence[Union[DataLoader, dict]],
                 stages: Sequence[dict]) -> None:
        self._runner = runner
        if len(dataloader) != len(stages):
            raise AssertionError(
                'number of the dataloaders must equal to '
                f'the number of stages, bug got {len(dataloader)} and'
                f' {len(stages)}.')
        self._loop_stages = []
        for dataloader, loop_cfg in zip(dataloader, stages):
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(runner=runner, dataloader=dataloader))
            self._loop_stages.append(loop)
        self._cur_loop = self._loop_stages[0]
        self._max_epochs = sum((loop.max_epochs for loop in self._loop_stages))
        self._max_iters = sum((loop.max_iters for loop in self._loop_stages))
        self._previous_epochs = 0
        self._previous_iters = 0

    @property
    def dataloader(self):
        return self._cur_loop.dataloader

    @property
    def max_epochs(self) -> int:
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self) -> int:
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self) -> int:
        """int: Current epoch."""
        return self._previous_epochs + self._cur_loop.epoch

    @property
    def iter(self) -> int:
        """int: Current iteration."""
        return self._previous_iters + self._cur_loop.iter

    def run(self) -> torch.nn.Module:
        """Execute loop."""
        for loop in self._loop_stages:
            self._cur_loop = loop
            model = loop.run()
            self._previous_epochs += loop.epoch
            self._previous_iters += loop.iter
        return model
