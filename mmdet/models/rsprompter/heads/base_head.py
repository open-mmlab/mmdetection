# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple

from mmengine.model import BaseModule
from mmengine.structures import BaseDataElement


class BaseHead(BaseModule, metaclass=ABCMeta):
    """Base head.

    Args:
        init_cfg (dict, optional): The extra init config of layers.
            Defaults to None.
    """

    def __init__(self, init_cfg: Optional[dict] = None):
        super(BaseHead, self).__init__(init_cfg=init_cfg)

    @abstractmethod
    def loss(self, feats: Tuple, data_samples: List[BaseDataElement]):
        """Calculate losses from the extracted features.

        Args:
            feats (tuple): The features extracted from the backbone.
            data_samples (List[BaseDataElement]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        pass

    @abstractmethod
    def predict(self,
                feats: Tuple,
                data_samples: Optional[List[BaseDataElement]] = None):
        """Predict results from the extracted features.

        Args:
            feats (tuple): The features extracted from the backbone.
            data_samples (List[BaseDataElement], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[BaseDataElement]: A list of data samples which contains the
            predicted results.
        """
        pass
