# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch

try:
    import mmcls
    from mmcls.models.classifiers import ImageClassifier
except ImportError:
    mmcls = None
    ImageClassifier = object

from mmdet.registry import MODELS
from mmdet.structures import ReIDDataSample


@MODELS.register_module()
class BaseReID(ImageClassifier):
    """Base model for re-identification."""

    def __init__(self, *args, **kwargs):
        if mmcls is None:
            raise RuntimeError('Please run "pip install openmim" and '
                               'run "mim install mmcls>=1.0.0rc0" tp '
                               'install mmcls first.')
        super().__init__(*args, **kwargs)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[ReIDDataSample]] = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
          tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`ReIDDataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, H, W) or (N, T, C, H, W).
            data_samples (List[ReIDDataSample], optional): The annotation
                data of every sample. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of
              :obj:`ReIDDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if len(inputs.size()) == 5:
            assert inputs.size(0) == 1
            inputs = inputs[0]
        return super().forward(inputs, data_samples, mode)
