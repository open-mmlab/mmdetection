# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in mmdetection."""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from mmengine.config import ConfigDict
from mmengine.data import InstanceData, PixelData

from ..bbox.samplers import SamplingResult
from ..data_structures import DetDataSample

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

InstanceList = List[InstanceData]
OptInstanceList = Optional[InstanceList]

PixelList = List[PixelData]
OptPixelList = Optional[PixelList]

SampleList = List[DetDataSample]
OptSampleList = Optional[SampleList]

SamplingResultList = List[SamplingResult]

OptSamplingResultList = Optional[SamplingResultList]

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

RangeType = Sequence[Tuple[int, int]]
