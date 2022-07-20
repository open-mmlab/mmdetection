# Copyright (c) OpenMMLab. All rights reserved.
from collections import Sequence

import torch

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_project
from .base import BaseDetector


@MODELS.register_module()
class SemiBaseDetector(BaseDetector):

    def __init__(self,
                 detector,
                 semi_train_cfg=None,
                 semi_test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.teacher = MODELS.build(detector)
        self.student = MODELS.build(detector)
        self.semi_train_cfg = semi_train_cfg
        self.semi_test_cfg = semi_test_cfg
        self.freeze(self.teacher)

    @staticmethod
    def freeze(model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def loss(self, multi_batch_inputs, multi_batch_data_samples):
        losses = dict()
        losses.update(**self.gt_loss(multi_batch_inputs['sup'],
                                     multi_batch_data_samples['sup']))
        origin_pseudo_instances, batch_info = self.get_pseudo_instances(
            multi_batch_inputs['unsup_teacher'],
            multi_batch_data_samples['unsup_teacher'])
        multi_batch_data_samples[
            'unsup_student'] = self.project_pseudo_instances(
                origin_pseudo_instances,
                multi_batch_data_samples['unsup_student'])
        losses.update(**self.pseudo_loss(
            multi_batch_inputs['unsup_student'],
            multi_batch_data_samples['unsup_student'], batch_info))
        return losses

    def gt_loss(self, batch_inputs, batch_data_samples):
        gt_loss = {
            'sup_' + k: v
            for k, v in self.weight(
                self.student.loss(batch_inputs, batch_data_samples),
                self.semi_train_cfg.get('sup_weight', 1.)).items()
        }
        return gt_loss

    def pseudo_loss(self, batch_inputs, batch_data_samples, batch_info):
        score_thr = self.semi_train_cfg.get('score_thr', 0.9)
        batch_data_samples = self.filter_pseudo_instances(
            batch_data_samples, score_thr)
        pseudo_instances_num = sum([
            len(data_samples.gt_instances)
            for data_samples in batch_data_samples
        ])
        unsup_weight = self.semi_train_cfg.get(
            'unsup_weight', 1.) if pseudo_instances_num > 0 else 0.
        pseudo_loss = {
            'unsup_' + k: v
            for k, v in self.weight(
                self.student.loss(batch_inputs, batch_data_samples),
                unsup_weight).items()
        }
        return pseudo_loss

    @staticmethod
    def weight(losses, weight):
        for name, loss in losses.items():
            if 'loss' in name:
                if isinstance(loss, Sequence):
                    losses[name] = [item * weight for item in loss]
                else:
                    losses[name] = loss * weight
        return losses

    def filter_pseudo_instances(self, batch_data_samples, score_thr):
        for data_samples in batch_data_samples:
            pseudo_bboxes = data_samples.gt_instances.bboxes
            instance_num = pseudo_bboxes.shape[0]
            if instance_num == 0:
                continue

            w = pseudo_bboxes[:, 2] - pseudo_bboxes[:, 0]
            h = pseudo_bboxes[:, 3] - pseudo_bboxes[:, 1]
            valid_size_flag = (
                w > self.semi_train_cfg.min_pseudo_bbox_wh[0]) & (
                    h > self.semi_train_cfg.min_pseudo_bbox_wh[1])
            valid_score_flag = data_samples.gt_instances.scores > score_thr
            valid_flag = valid_size_flag & valid_score_flag
            data_samples.gt_instances = data_samples.gt_instances.new(
                scores=data_samples.gt_instances.scores[valid_flag],
                bboxes=data_samples.gt_instances.bboxes[valid_flag],
                labels=data_samples.gt_instances.labels[valid_flag])
        return batch_data_samples

    def get_pseudo_instances(self, batch_inputs, batch_data_samples):
        results_list = self.teacher.predict(batch_inputs, batch_data_samples)
        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = data_samples.gt_instances.new(
                scores=results.pred_instances.scores,
                bboxes=results.pred_instances.bboxes,
                labels=results.pred_instances.labels)
        return batch_data_samples, None

    def project_pseudo_instances(self, batch_pseudo_instances,
                                 batch_data_samples):
        for pseudo_instances, data_samples in zip(batch_pseudo_instances,
                                                  batch_data_samples):
            pseudo_instances.gt_instances.bboxes = bbox_project(
                pseudo_instances.gt_instances.bboxes,
                torch.tensor(data_samples.homography_matrix).to(
                    self.data_preprocessor.device), data_samples.img_shape)
            data_samples.gt_instances = pseudo_instances.gt_instances
        return batch_data_samples

    def predict(self, batch_inputs, batch_data_samples):
        if self.semi_test_cfg.get('infer_on', None) == 'teacher':
            return self.teacher(
                batch_inputs, batch_data_samples, mode='predict')
        else:
            return self.student(
                batch_inputs, batch_data_samples, mode='predict')

    def _forward(self, batch_inputs, batch_data_samples):
        return self.student(batch_inputs, batch_data_samples, mode='tensor')

    def extract_feat(self, batch_inputs):
        return self.student.extract_feat(batch_inputs)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any([
                'student' in key or 'teacher' in key
                for key in state_dict.keys()
        ]):
            keys = list(state_dict.keys())
            state_dict.update({'teacher.' + k: state_dict[k] for k in keys})
            state_dict.update({'student.' + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
