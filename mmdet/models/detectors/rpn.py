# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from inspect import signature

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector


@MODELS.register_module()
class RPN(SingleStageDetector):
    """Implementation of Region Proposal Network.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
<<<<<<< HEAD
                 backbone,
                 neck,
                 rpn_head,
                 train_cfg,
                 test_cfg,
                 init_cfg=None):
        super(RPN, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck) if neck is not None else None
        rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
=======
                 backbone: ConfigType,
                 neck: ConfigType,
                 rpn_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        rpn_train_cfg = train_cfg['rpn'] if train_cfg is not None else None
        rpn_head_num_classes = rpn_head.get('num_classes', 1)
        if rpn_head_num_classes != 1:
            warnings.warn('The `num_classes` should be 1 in RPN, but get '
                          f'{rpn_head_num_classes}, please set '
                          'rpn_head.num_classes = 1 in your config file.')
            rpn_head.update(num_classes=1)
>>>>>>> mmdetection/main
        rpn_head.update(train_cfg=rpn_train_cfg)
        rpn_head.update(test_cfg=test_cfg['rpn'])
        self.bbox_head = MODELS.build(rpn_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)

        # set cat_id of gt_labels to 0 in RPN
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        losses = self.bbox_head.loss(x, rpn_data_samples)
        return losses
<<<<<<< HEAD

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        x = self.extract_feat(img)
        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        if rescale:
            for proposals, meta in zip(proposal_list, img_metas):
                proposals[:, :4] /= proposals.new_tensor(meta['scale_factor'])
        if torch.onnx.is_in_onnx_export():
            return proposal_list

        return [proposal.cpu().numpy() for proposal in proposal_list]

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        proposal_list = self.rpn_head.aug_test_rpn(
            self.extract_feats(imgs), img_metas)
        if not rescale:
            for proposals, img_meta in zip(proposal_list, img_metas[0]):
                img_shape = img_meta['img_shape']
                scale_factor = img_meta['scale_factor']
                flip = img_meta['flip']
                flip_direction = img_meta['flip_direction']
                proposals[:, :4] = bbox_mapping(proposals[:, :4], img_shape,
                                                scale_factor, flip,
                                                flip_direction)
        return [proposal.cpu().numpy() for proposal in proposal_list]

    def show_result(self, data, result, top_k=20, **kwargs):
        """Show RPN proposals on the image.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            top_k (int): Plot the first k bboxes only
               if set positive. Default: 20

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if kwargs is not None:
            kwargs['colors'] = 'green'
            sig = signature(mmcv.imshow_bboxes)
            for k in list(kwargs.keys()):
                if k not in sig.parameters:
                    kwargs.pop(k)
        mmcv.imshow_bboxes(data, result, top_k=top_k, **kwargs)
=======
>>>>>>> mmdetection/main
