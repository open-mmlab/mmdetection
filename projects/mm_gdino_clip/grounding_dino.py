# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from mmdet.models.detectors import GroundingDINO

task_map={'OD': 0, 'REC': 0, 'VG': 1}

@MODELS.register_module()
class GroundingDINOV2(GroundingDINO):

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        tasks=[data_samples.dataset_mode for data_samples in batch_data_samples]
        tasks= [task_map[task] for task in tasks]
        assert len(set(tasks)) == 1, 'Only support one task in one batch, but got {}'.format(tasks)

        if tasks[0]==1:
            # VG
            return super().loss(batch_inputs, batch_data_samples)
        else:
            # OD=REC
            text_prompts = [
                data_samples.text for data_samples in batch_data_samples
            ]

            gt_labels = [
                data_samples.gt_instances.labels
                for data_samples in batch_data_samples
            ]

            text_dict = self.language_model(text_prompts, task='OD')
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])



