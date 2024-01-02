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

task_map = {'REC': 0, 'VG': 1}


@MODELS.register_module()
class GroundingDINOV2(GroundingDINO):

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        tasks = [data_samples.dataset_mode for data_samples in batch_data_samples]
        tasks = [task_map[task] for task in tasks]
        assert len(set(tasks)) == 1, 'Only support one task in one batch, but got {}'.format(tasks)

        if tasks[0] == 1:
            # VG
            return super().loss(batch_inputs, batch_data_samples)
        else:
            # REC
            text_prompts = [
                data_samples.text for data_samples in batch_data_samples
            ]

            text_dict = self.language_model(text_prompts, task='REC')
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

            for i, data_samples in enumerate(batch_data_samples):
                # for calc BinaryFocalLossCost
                text_token_mask = text_dict['text_token_mask'][i]
                data_samples.gt_instances.text_token_mask = \
                    text_token_mask.unsqueeze(0).repeat(
                        len(data_samples.gt_instances), 1)

            visual_features = self.extract_feat(batch_inputs)
            head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                        batch_data_samples)

            losses = self.bbox_head.loss(
                **head_inputs_dict, batch_data_samples=batch_data_samples)
            return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        # only od eval for now
        text_prompts = [data_samples.text for data_samples in batch_data_samples]
        text_prompts = text_prompts[0]

        visual_feats = self.extract_feat(batch_inputs)

        text_dict = self.language_model([text_prompts], task='REC')
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(
                text_dict['embedded'])
        head_inputs_dict = self.forward_transformer(
            visual_feats, text_dict, batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        for data_sample, pred_instances in zip(
                batch_data_samples, results_list):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    label_names.append(text_prompts[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
