import pyclipper
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, xavier_init
from torch.onnx import is_in_onnx_export

from mmdet.core import multi_apply, force_fp32
from mmdet.ops import batched_nms
from ...core.utils.misc import topk
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from .rpn_test_mixin import RPNTestMixin

import numpy as np
import cv2


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=has_bias
    )


def conv3x3_bn_relu(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def vatti_clipping(contours, ratio):
    clipped_contours = []
    for contour in contours:
        contour = np.array(contour).reshape(-1, 2)
        length = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        d = area * ratio / max(length, 1)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(contour, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        clipped = pco.Execute(-d)
        for s in clipped:
            s = np.array(s).reshape(-1, 2)
            clipped_contours.append(s)
    return clipped_contours


def clip_contours(contours):
    return vatti_clipping(contours, (1 - np.power(0.4, 2)))


def unclip_contours(contours):
    return vatti_clipping(contours, -3.0)


@HEADS.register_module()
class SPNHead(nn.Module):

    def __init__(self, in_channels, feat_channels, train_cfg, test_cfg, loss_mask):
        super(SPNHead, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_mask = build_loss(loss_mask)

        self.prob = nn.Sequential(
            conv3x3_bn_relu(self.in_channels, self.feat_channels, 1),
            nn.ConvTranspose2d(self.feat_channels, self.feat_channels, 2, 2),
            nn.BatchNorm2d(self.feat_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.feat_channels, 1, 2, 2),
            nn.Sigmoid(),
        )

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, )):
                xavier_init(m, distribution='uniform')

    def forward_single(self, feats):
        return tuple([self.prob(feats)])

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        gt_masks = kwargs['gt_masks']
        outs = self(x)
        loss_inputs = outs + (gt_masks, img_metas)
        losses = self.loss(*loss_inputs)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, cfg=proposal_cfg)
            return losses, proposal_list

    def simple_test_rpn(self, x, img_metas):
        rpn_outs = self(x)
        proposal_list = self.get_bboxes(*rpn_outs, cfg=None)
        return proposal_list

    def _forward_train(self, preds, targets, image_shapes):
        # Segmentation map must be transformed into boxes for detection.
        # sampled into a training batch.
        with torch.no_grad():
            boxes = self.box_selector_train(preds, image_shapes, targets)
        loss_seg = self.loss_evaluator(preds, targets)
        losses = {"loss_seg": loss_seg}
        return boxes, losses

    def _forward_test(self, preds, image_shapes):
        # torch.cuda.synchronize()
        # start_time = time.time()
        boxes, rotated_boxes, polygons, scores = self.box_selector_test(
            preds, image_shapes)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('post time:', end_time - start_time)
        seg_results = {'rotated_boxes': rotated_boxes,
                       'polygons': polygons, 'preds': preds, 'scores': scores}
        return boxes, seg_results

    def get_targets(self, mask_pred, gt_masks):
        united_masks = []
        for masks in gt_masks:
            assert len(masks[0]), '????'
            final_mask = np.zeros_like(masks[0])
            clipped_contours = []
            for mask in masks:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = clip_contours(contours)
                clipped_contours.extend(contours)

            final_mask = np.squeeze(final_mask, axis=0)
            cv2.drawContours(final_mask, clipped_contours, -1, 1, -1)
            united_masks.append(final_mask)

        mask_targets_per_level = []
        for level_idx, level_pred in enumerate(mask_pred):
            mask_targets = [cv2.resize(
                mask, level_pred.shape[-2:][::-1], cv2.INTER_NEAREST) for mask in united_masks]
            mask_targets = [np.expand_dims(mask, axis=(0, 1))
                            for mask in mask_targets]
            mask_targets = [torch.tensor(mask, device=level_pred.device, dtype=level_pred.dtype)
                            for mask in mask_targets]
            mask_targets_per_level.append(torch.cat(mask_targets))

        return mask_targets_per_level

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, gt_masks, img_metas):
        mask_targets = self.get_targets(mask_pred, gt_masks)

        if 0:
            cpu_targets = [np.squeeze(mask.cpu().numpy().astype(np.uint8))
                        for mask in mask_targets]
            cpu_preds = [c.detach().cpu().numpy() for c in mask_pred]
            for m, cc in zip(cpu_targets, cpu_preds):
                for mb, c in zip(m, cc):
                    cv2.imshow('target', np.squeeze(mb) * 255)
                    cv2.imshow('preds', (np.squeeze(c) * 255).astype(np.uint8))
                    cv2.imshow('thresdolded', ((np.squeeze(c) > 0.5) * 255).astype(np.uint8))
                    cv2.waitKey(1)

        mask_loss = sum(self.loss_mask(pred, target)
                        for pred, target in zip(mask_pred, mask_targets))
        loss = {'loss_rpn_mask': mask_loss}
        return loss

    def get_bboxes(self, mask_preds, cfg, rescale=False):
        batch_size = mask_preds[0].shape[0]
        proposals_list = []

        thr = 0.5

        for level_idx, level_pred in enumerate(mask_preds):
            for mask_idx_in_batch, mask_in_batch in enumerate(level_pred):
                pred_cpu = np.squeeze(mask_in_batch.detach().cpu().numpy())
                mask_cpu = (pred_cpu > thr).astype(np.uint8)
                contours, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) > cfg.nms_post:
                    print(len(contours))
                    contours = sorted(contours, key=lambda x: cv2.contourArea(x))[:cfg.nms_post]

                contours = unclip_contours(contours)
        
                boxes = []
                labels = []

                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    boxes.append(torch.tensor(np.array([[x, y, x + w, y + h, 1.0]]), device=mask_preds[0].device,
                                              dtype=torch.float))
                    labels.append(torch.tensor(np.array([[1]])))

                # mask_cpu = np.zeros_like(mask_cpu)
                # for c in contours:
                #     cv2.rectangle(mask_cpu, (x, y), (x+w, y+h), 1)
                # cv2.drawContours(mask_cpu, contours, -1, 1, -1)
                # cv2.imshow("res", mask_cpu * 255)
                # cv2.waitKey(0)

                if boxes:
                    boxes = torch.cat(boxes)
                    labels = torch.cat(labels)
                else:
                    boxes = torch.zeros(
                        (0, 5), device=mask_preds[0].device, dtype=torch.float)
                    labels = torch.zeros(
                        (0, 1), device=mask_preds[0].device, dtype=torch.int)

                proposals_list.append(boxes)
                    

        return proposals_list
