# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from mmengine.data import InstanceData

from mmdet.core import bbox2roi, bbox_project
from mmdet.models.losses import accuracy
from mmdet.registry import MODELS
from ..utils.misc import unpack_gt_instances
from .semi_base import SemiBaseDetector


@MODELS.register_module()
class SoftTeacher(SemiBaseDetector):

    def __init__(self,
                 detector,
                 semi_train_cfg=None,
                 semi_test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def pseudo_loss(self, batch_inputs, batch_data_samples, batch_info):
        x = self.student.extract_feat(batch_inputs)

        losses = {}
        rpn_losses, rpn_results_list = self.unsup_rpn_loss(
            x, batch_data_samples)
        losses.update(rpn_losses)
        losses.update(
            self.unsup_rcnn_cls_loss(x, rpn_results_list, batch_data_samples,
                                     batch_info))
        losses.update(
            self.unsup_rcnn_reg_loss(x, rpn_results_list, batch_data_samples))

        pseudo_instances_num = sum([
            len(data_samples.gt_instances)
            for data_samples in batch_data_samples
        ])
        unsup_weight = self.semi_train_cfg.get(
            'unsup_weight', 1.) if pseudo_instances_num > 0 else 0.

        pseudo_loss = {
            'unsup_' + k: v
            for k, v in self.weight(losses, unsup_weight).items()
        }
        return pseudo_loss

    def get_pseudo_instances(self, batch_inputs, batch_data_samples):
        assert self.teacher.with_bbox, 'Bbox head must be implemented.'
        x = self.teacher.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.teacher.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.teacher.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=False)

        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results

        score_thr = self.semi_train_cfg.get('pseudo_label_initial_score_thr',
                                            0.5)
        batch_data_samples = self.filter_pseudo_instances(
            batch_data_samples, score_thr)

        reg_uncs_list = self.compute_uncertainty_with_aug(
            x, batch_data_samples)

        for data_samples, reg_uncs in zip(batch_data_samples, reg_uncs_list):
            data_samples.gt_instances['reg_uncs'] = reg_uncs
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.tensor(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)

        batch_info = {'feat': x, 'img_shape': [], 'homography_matrix': []}
        for data_samples in batch_data_samples:
            batch_info['img_shape'].append(data_samples.img_shape)
            batch_info['homography_matrix'].append(
                torch.tensor(data_samples.homography_matrix).to(
                    self.data_preprocessor.device))
        return batch_data_samples, batch_info

    def unsup_rpn_loss(self, x, batch_data_samples):
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        for data_samples in rpn_data_samples:
            if data_samples.gt_instances.bboxes.shape[0] > 0:
                data_samples.gt_instances = data_samples.gt_instances[
                    data_samples.gt_instances.scores >
                    self.semi_train_cfg.rpn_pseudo_thr]
            else:
                continue

        proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                                  self.student.test_cfg.rpn)

        # set cat_id of gt_labels to 0 in RPN
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        rpn_losses, rpn_results_list = self.student.rpn_head.loss_and_predict(
            x, rpn_data_samples, proposal_cfg=proposal_cfg)
        for key in rpn_losses.keys():
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        return rpn_losses, rpn_results_list

    def unsup_rcnn_cls_loss(self, x, unsup_rpn_results_list,
                            batch_data_samples, batch_info):
        rpn_results_list = copy.deepcopy(unsup_rpn_results_list)
        cls_data_samples = copy.deepcopy(batch_data_samples)
        for data_samples in cls_data_samples:
            if data_samples.gt_instances.bboxes.shape[0] > 0:
                data_samples.gt_instances = data_samples.gt_instances[
                    data_samples.gt_instances.scores >
                    self.semi_train_cfg.cls_pseudo_thr]
            else:
                continue
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.student.roi_head.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        selected_bboxes = [res.priors for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(x, rois)
        labels, label_weights, bbox_targets, bbox_weights = \
            self.student.roi_head.bbox_head.get_targets(
                sampling_results, self.student.train_cfg.rcnn)

        selected_results_list = []
        for bboxes, data_samples, homography_matrix, img_shape in zip(
                selected_bboxes, batch_data_samples,
                batch_info['homography_matrix'], batch_info['img_shape']):
            selected_results_list.append(
                InstanceData(
                    bboxes=bbox_project(
                        bboxes, homography_matrix @ torch.tensor(
                            data_samples.homography_matrix).inverse().to(
                                self.data_preprocessor.device), img_shape)))

        with torch.no_grad():
            results_list = self.teacher.roi_head.predict_bbox(
                batch_info['feat'],
                [data_samples.metainfo for data_samples in batch_data_samples],
                selected_results_list,
                rcnn_test_cfg=None,
                rescale=False)
            bg_score = torch.cat(
                [results.scores[:, -1] for results in results_list])
            neg_inds = labels == self.student.roi_head.bbox_head.num_classes
            label_weights[neg_inds] = bg_score[neg_inds].detach()

        cls_score = bbox_results['cls_score']
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.student.roi_head.bbox_head.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override='none')
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.student.roi_head.bbox_head.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        losses['loss_cls'] = losses['loss_cls'].sum() / max(
            label_weights.sum(), 1.0)
        return losses

    def unsup_rcnn_reg_loss(self, x, unsup_rpn_results_list,
                            batch_data_samples):
        rpn_results_list = copy.deepcopy(unsup_rpn_results_list)
        reg_data_samples = copy.deepcopy(batch_data_samples)
        for data_samples in reg_data_samples:
            if data_samples.gt_instances.bboxes.shape[0] > 0:
                data_samples.gt_instances = data_samples.gt_instances[
                    data_samples.gt_instances.reg_uncs <
                    self.semi_train_cfg.reg_pseudo_thr]
            else:
                continue
        roi_losses = self.student.roi_head.loss(x, rpn_results_list,
                                                reg_data_samples)
        return {'loss_bbox': roi_losses['loss_bbox']}

    def compute_uncertainty_with_aug(self, x, batch_data_samples):
        auged_results_list = self.aug_box(batch_data_samples,
                                          self.semi_train_cfg.jitter_times,
                                          self.semi_train_cfg.jitter_scale)
        # flatten
        auged_results_list = [
            InstanceData(bboxes=auged.reshape(-1, auged.shape[-1]))
            for auged in auged_results_list
        ]

        self.teacher.roi_head.test_cfg = None
        results_list = self.teacher.roi_head.predict(
            x, auged_results_list, batch_data_samples, rescale=False)
        self.teacher.roi_head.test_cfg = self.teacher.test_cfg.rcnn

        reg_channel = max(
            [results.bboxes.shape[-1] for results in results_list]) // 4
        bboxes = [
            results.bboxes.reshape(self.semi_train_cfg.jitter_times, -1,
                                   results.bboxes.shape[-1])
            if results.bboxes.numel() > 0 else results.bboxes.new_zeros(
                self.semi_train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for results in results_list
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel,
                             4)[torch.arange(bbox.shape[0]), label]
                for bbox, label in zip(bboxes, labels)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel,
                            4)[torch.arange(unc.shape[0]), label]
                for unc, label in zip(box_unc, labels)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0)
                     for bbox in bboxes]
        box_unc = [
            torch.mean(
                unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4), dim=-1)
            if wh.numel() > 0 else unc for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    @staticmethod
    def aug_box(batch_data_samples, times, frac):

        def _aug_single(box):
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2,
                                                          2).reshape(-1, 4))
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device) *
                aug_scale[None, ...])
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]],
                dim=-1)

        return [
            _aug_single(data_samples.gt_instances.bboxes)
            for data_samples in batch_data_samples
        ]
