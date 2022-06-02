# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in mmdetection."""
from typing import List, Optional, Union

from mmengine.config import ConfigDict
from mmengine.data import InstanceData

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

SampleList = List[DetDataSample]
OptSampleList = Optional[SampleList]

SamplingResultList = List[SamplingResult]
