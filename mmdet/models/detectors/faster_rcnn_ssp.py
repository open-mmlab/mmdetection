import torch
import torch.nn as nn
import cv2
import numpy as np
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.models.module.heatmap import Heatmap


class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        self.iou_calculator = build_iou_calculator(dict(type='BboxOverlaps2D'))

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)
        self.att_loss = Heatmap()

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        # super(TwoStageDetector, self).init_weights(pretrained)
        super(TwoStageDetector, self).init_weights()
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x, att = self.neck(x)
            if self.training:
                return x, att
            else:
                return x
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x, att = self.extract_feat(img)
        gt_reg = self.att_loss.target(att, gt_bboxes)
        loss_att = self.att_loss.loss(reg_pred=att, reg_gt=gt_reg)

        losses = dict()
        losses.update(loss_att)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)

        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    @staticmethod
    def show_img(imgs, window_names=None, wait_time_ms=0, is_merge=False, row_col_num=(1, -1)):
        """
        Displays an image or a list of images in specified windows or self-initiated windows.
        You can also control display wait time by parameter 'wait_time_ms'.
        Additionally, this function provides an optional parameter 'is_merge' to
        decide whether to display all imgs in a particular window 'merge'.
        Besides, parameter 'row_col_num' supports user specified merge format.
        Notice, specified format must be greater than or equal to imgs number.

        :param imgs: numpy.ndarray or list.
        :param window_names: specified or None, if None, function will create different windows as '1', '2'.
        :param wait_time_ms: display wait time.
        :param is_merge: whether to merge all images.
        :param row_col_num: merge format. default is (1, -1), image will line up to show.
                            example=(2, 5), images will display in two rows and five columns.
        """
        if not isinstance(imgs, list):
            imgs = [imgs]

        if window_names is None:
            window_names = list(range(len(imgs)))
        else:
            if not isinstance(window_names, list):
                window_names = [window_names]
            assert len(imgs) == len(window_names), 'window names does not match images!'

        if is_merge:
            merge_imgs = TwoStageDetector.merge_imgs(imgs, row_col_num)

            cv2.namedWindow('merge', 0)
            cv2.imshow('merge', merge_imgs)
        else:
            for img, win_name in zip(imgs, window_names):
                if img is None:
                    continue
                win_name = str(win_name)
                cv2.namedWindow(win_name, 0)
                cv2.imshow(win_name, img)

        cv2.waitKey(wait_time_ms)

    @staticmethod
    def merge_imgs(imgs, row_col_num):
        """
        Merges all input images as an image with specified merge format.

        :param imgs : img list
        :param row_col_num : number of rows and columns displayed
        :return img : merges img
        """

        # from ..visualtools import random_color

        length = len(imgs)
        row, col = row_col_num

        assert row > 0 or col > 0, 'row and col cannot be negative at same time!'
        # color = random_color(rgb=True).astype(np.float64)
        color = (0, 0, 255)
        for img in imgs:
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color)

        if row_col_num[1] < 0 or length < row:
            merge_imgs = np.hstack(imgs)
        elif row_col_num[0] < 0 or length < col:
            merge_imgs = np.vstack(imgs)
        else:
            assert row * col >= length, 'Imgs overboundary, not enough windows to display all imgs!'

            fill_img_list = [np.zeros(imgs[0].shape, dtype=np.uint8)] * (row * col - length)
            imgs.extend(fill_img_list)
            merge_imgs_col = []
            for i in range(row):
                start = col * i
                end = col * (i + 1)
                merge_col = np.hstack(imgs[start: end])
                merge_imgs_col.append(merge_col)

            merge_imgs = np.vstack(merge_imgs_col)

        return merge_imgs

    @staticmethod
    # 可视化显示相关
    def show_bbox(image, bboxs_list, color=None,
                  thickness=1, font_scale=0.3, wait_time_ms=0, names=None,
                  is_show=True, is_without_mask=False):
        """
        Visualize bbox in object detection by drawing rectangle.

        :param image: numpy.ndarray.
        :param bboxs_list: list: [pts_xyxy, prob, id]: label or prediction.
        :param color: tuple.
        :param thickness: int.
        :param fontScale: float.
        :param wait_time_ms: int
        :param names: string: window name
        :param is_show: bool: whether to display during middle process
        :return: numpy.ndarray
        """
        # from ..visualtools import random_color
        assert image is not None
        font = cv2.FONT_HERSHEY_SIMPLEX
        image_copy = image.copy()
        colorss = [(0, 255, 0), (0, 0, 255)]
        for id, bbox_list in enumerate(bboxs_list):
            colors = colorss[id]
            for bbox in bbox_list:
                if len(bbox) == 5:
                    txt = '{:.3f}'.format(bbox[4])
                elif len(bbox) == 6:
                    txt = 'p={:.3f},id={:.3f}'.format(bbox[4], bbox[5])
                bbox_f = np.array(bbox[:4], np.int32)[0]
                # if color is None:
                #     colors = random_color(rgb=True).astype(np.float64)
                # else:
                #     colors = color
                # colors = (0, 255, 0)

                if not is_without_mask:
                    image_copy = cv2.rectangle(image_copy, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors,
                                               thickness)
                else:
                    mask = np.zeros_like(image_copy, np.uint8)
                    mask1 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, -1)
                    mask = np.zeros_like(image_copy, np.uint8)
                    mask2 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, thickness)
                    mask2 = cv2.addWeighted(mask1, 0.5, mask2, 8, 0.0)
                    image_copy = cv2.addWeighted(image_copy, 1.0, mask2, 0.6, 0.0)
                if len(bbox) == 5 or len(bbox) == 6:
                    cv2.putText(image_copy, txt, (bbox_f[0], bbox_f[1] - 2),
                                font, font_scale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
        if is_show:
            TwoStageDetector.show_img(image_copy, names, wait_time_ms)
        return image_copy


@DETECTORS.register_module()
class FasterSSPNet(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(FasterSSPNet, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
