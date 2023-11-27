# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict

import torch
from torch import Tensor

from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.registry import MODELS
from mmdet.structures import SampleList


@MODELS.register_module()
class OVDTwoStageDetector(TwoStageDetector):

    def __init__(self, batch2ovd=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch2ovd = dict() if batch2ovd is None else batch2ovd
        # mapping from batch name to ovd name

    def run_ovd(self, x, inputs, data_samples, ovd_name):
        losses = dict()
        if self.with_rpn:
            with torch.no_grad():
                rpn_results_list = self.rpn_head_predict(x, data_samples)
        else:
            assert data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in data_samples
            ]
        if isinstance(ovd_name, str):
            ovd_name = [ovd_name]
        for _ovd_name in ovd_name:
            losses.update(
                self.roi_head.run_ovd(x, data_samples, rpn_results_list,
                                      _ovd_name, inputs))
        return losses

    def rpn_head_predict(self, x, batch_data_samples):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self.rpn_head(x)
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        predictions = self.rpn_head.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            cfg=proposal_cfg,
            rescale=False)
        return predictions

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        if not isinstance(multi_batch_inputs, dict):
            multi_batch_inputs = dict(det_batch=multi_batch_inputs)
            multi_batch_data_samples = dict(det_batch=multi_batch_data_samples)

        multi_batch_features = {
            k: self.extract_feat(v)
            for k, v in multi_batch_inputs.items()
        }
        losses = self.det_loss(
            multi_batch_features.get('det_batch'),
            multi_batch_data_samples.get('det_batch'))

        for batch_name, ovd_name in self.batch2ovd.items():
            batch_inputs = multi_batch_inputs.get(batch_name)
            batch_data_samples = multi_batch_data_samples.get(batch_name)
            batch_features = multi_batch_features.get(batch_name)
            loss_ovd = self.run_ovd(batch_features, batch_inputs,
                                    batch_data_samples, ovd_name)
            for k, v in loss_ovd.items():
                losses.update({k + f'_{batch_name}': v})
        return losses

    def det_loss(self, x, batch_data_samples):
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses
