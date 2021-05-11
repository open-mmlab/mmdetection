import torch

from ...core.post_processor.bbox_nms import NMS
from ...core.post_processor.builder import ComposePostProcess
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 aug_bbox_post_processes=[
                     dict(type='MergeResults'),
                     dict(
                         type='NaiveNMS',
                         iou_threshold=0.5,
                         class_agnostic=False,
                         max_num=100)
                 ]):
        super(SingleStageDetector, self).__init__(init_cfg)

        if aug_bbox_post_processes is not None:
            self.aug_bbox_post_processes = ComposePostProcess(
                aug_bbox_post_processes)
        else:
            self.aug_bbox_post_processes = ComposePostProcess[dict(
                type='MergeResults')]
        backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape
        if torch.onnx.is_in_onnx_export():
            # get shape as tensor
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape

        results_list = self.bbox_head.get_bboxes(*outs, img_metas)
        if torch.onnx.is_in_onnx_export():
            return results_list

        return [results.export('bbox') for results in results_list]

    def aug_test(self, imgs, img_metas):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """

        assert len(img_metas[0]) == 1, 'AugTest only support batchsize == 1'
        feats = self.extract_feats(imgs)
        return self.aug_test_bboxes(feats, img_metas)

    def aug_test_bboxes(self, feats, img_metas):

        # remove the nms op from head at first iteration
        if not hasattr(self, 'remove_head_nms'):
            nms_op = None
            head_post_processes = \
                self.bbox_head.bbox_post_processes.process_list
            for index, operation in enumerate(head_post_processes):
                if isinstance(operation, NMS):
                    break
            # Some heads do not hav nms, such as detr
            if nms_op:
                head_post_processes.pop(index)
            self.remove_head_nms = True

        aug_resutls_list = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x)
            results = self.bbox_head.get_bboxes(*outs, img_meta)[0]
            aug_resutls_list.append(results)

        aug_resutls_list = self.aug_bbox_post_processes(aug_resutls_list)
        return [results.export('bbox') for results in aug_resutls_list]
