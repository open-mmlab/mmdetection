# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
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

    def loss(self, multi_batch_inputs, multi_batch_data_samples):
        losses = dict()
        losses.update(**self.gt_loss(multi_batch_inputs['sup'],
                                     multi_batch_data_samples['sup']))
        multi_batch_data_samples[
            'unsup_student'] = self.update_pseudo_instances(
                multi_batch_inputs['unsup_teacher'],
                multi_batch_data_samples['unsup_teacher'])
        losses.update(
            **self.pseudo_loss(multi_batch_inputs['unsup_student'],
                               multi_batch_data_samples['unsup_student']))
        return losses

    def gt_loss(self, batch_inputs, batch_data_samples):
        losses = self.student.loss(batch_inputs, batch_data_samples)
        w = self.semi_train_cfg.get('sup_weight', 1.)
        return {'sup_' + k: v * w for k, v in losses.items()}

    def pseudo_loss(self, batch_inputs, batch_data_samples):
        losses = self.student.loss(batch_inputs, batch_data_samples)
        w = self.semi_train_cfg.get('unsup_weight', 4.)
        return {'unsup_' + k: v * w for k, v in losses.items()}

    def update_pseudo_instances(self, batch_inputs, batch_data_samples):
        self.teacher.eval()
        results_list = self.teacher.predict(batch_inputs, batch_data_samples)
        return results_list

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
