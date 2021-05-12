from abc import ABCMeta, abstractmethod
from logging import warning

import torch
from mmcv import ConfigDict
from mmcv.runner import BaseModule

from mmdet.core.post_processor.builder import ComposePostProcess


def deploy_deprecate_warning(cfg):
    if 'deploy' not in cfg:
        warning('All deploy releated arguments has been moved to'
                ' a dict named deploy in test_cfg')
        cfg.deploy = ConfigDict(dict())
    if 'nms' in cfg:
        warning('When export the model to onnx, all nms related '
                'parameters has been moved to a dict `deploy` in '
                'test_cfg, and add a prefix `deploy_`, such as '
                '`max_per_img` to `deploy_max_per_img`')
        for n, v in cfg.nms.items():
            cfg.deploy[f'deploy_{n}'] = v

    deprecated_convert = {
        'max_per_img': 'deploy_max_per_img',
        'deploy_nms_pre': 'deploy_nms_pre',
        'score_thr': 'deploy_score_thr',
        # conf_thr only in YOLO
        'conf_thr': 'deploy_conf_thr'
    }
    for ori_name, convert_name in deprecated_convert.items():
        if ori_name in cfg:
            warning(f'Please specify {convert_name} instead of '
                    f' {ori_name} to a dict named '
                    ' `deploy` in test_cfg')
            cfg.deploy[convert_name] = cfg[ori_name]
    return cfg


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self,
                 bbox_post_processes=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.test_cfg:
            if self.test_cfg.get('score_thr', None):
                warning('The way to specify the score_thr has been'
                        'changed. Please specify it in '
                        'PreNMS in bbox_post_processes ')
            if self.test_cfg.get('nms', None):
                warning('The way to specify the type of mms and corresponding '
                        'iou_threshold has been'
                        'changed. Please specify it in '
                        ' bbox_post_processes ')
            if self.test_cfg.get('max_per_img', None):
                warning('The way to specify the max number of '
                        'bboxes after nms '
                        'has been changed. Please specify'
                        'it in bbox_post_processes ')
        if bbox_post_processes is not None:
            self.bbox_post_processes = ComposePostProcess(bbox_post_processes)
        else:
            raise RuntimeError(
                f'Please set post process for {self.__class__.__name__}')

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def onnx_trace(self, batch_mlvl_bboxes, batch_mlvl_scores, cfg):
        cfg = deploy_deprecate_warning(cfg)
        if cfg.deploy.get('with_nms', True):
            from mmdet.core.export import add_dummy_nms_for_onnx
            # ignore background class
            if not self.loss_cls.use_sigmoid:
                num_classes = batch_mlvl_scores.shape[2] - 1
                batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
            max_output_boxes_per_class = cfg.deploy.get(
                'max_output_boxes_per_class', 200)
            score_threshold = cfg.deploy.get('deploy_score_thr', 0.05)
            nms_pre = cfg.deploy.get('deploy_nms_pre', -1)
            iou_threshold = cfg.deploy.get('deploy_iou_threshold', 0.5)
            max_per_img = cfg.deploy.get('deploy_max_per_img', 100)

            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, max_per_img)
        else:
            if self.loss_cls.use_sigmoid:
                # Add a dummy background class to the backend when
                # using sigmoid remind that we set FG labels
                # to [0, num_class-1] since mmdet v2.0
                # BG cat_id: num_class
                padding = batch_mlvl_scores.new_zeros(
                    len(batch_mlvl_scores), batch_mlvl_scores.shape[1], 1)
                batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding],
                                              dim=-1)
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
            ]
            return det_results
