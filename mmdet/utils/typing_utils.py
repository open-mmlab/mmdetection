# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in mmdetection."""
from typing import List, Optional, Sequence, Tuple, Union

from mmengine.config import ConfigDict
from mmengine.structures import InstanceData, PixelData

# TODO: Need to avoid circular import with assigner and sampler
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

RangeType = Sequence[Tuple[int, int]]
