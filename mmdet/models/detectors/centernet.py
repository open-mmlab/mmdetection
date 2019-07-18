import cv2
import numpy as np
import torch
import torch.nn as nn

from ..registry import DATASETS, DETECTORS
from ..utils.ctdet_debugger import Debugger
from .two_stage import TwoStageDetector


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if isinstance(scale, torch.Tensor):
        scale = scale.cpu().squeeze().numpy()
    if isinstance(center, torch.Tensor):
        center = center.cpu().squeeze().numpy()
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    if isinstance(dst_w, torch.Tensor):
        dst_w = dst_w.cpu().squeeze().numpy()
    if isinstance(dst_h, torch.Tensor):
        dst_h = dst_h.cpu().squeeze().numpy()

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1),
                             topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([
        xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2, xs + wh[..., 0:1] / 2,
        ys + wh[..., 1:2] / 2
    ],
                       dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = []
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds.append(
                np.concatenate([
                    dets[i, inds, :4].astype(np.float32),
                    dets[i, inds, 4:5].astype(np.float32)
                ],
                               axis=1).tolist())
        ret.append(top_preds)
    return ret


@DETECTORS.register_module
class CenterNet(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(CenterNet, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        # self.loss = CtdetLoss()
        self.max_per_image = 100
        self.test_cfg = test_cfg
        if test_cfg:
            self.num_classes = test_cfg['num_classes']
            self.debugger = Debugger(
                dataset=DATASETS.get('Ctdet'),
                theme='black',
                num_classes=self.num_classes)

    def forward_train(self, img, img_meta, **kwargs):
        output = self.backbone(img)
        if self.rpn_head:
            output = self.rpn_head(output)

        losses = self.rpn_head.loss(output, **kwargs)

        return losses

    def post_process_test(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(dets.copy(), [meta['ctdet_c']],
                                  [meta['ctdet_s']], meta['ctdet_out_height'],
                                  meta['ctdet_out_width'], self.num_classes)
        for j in range(self.num_classes):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        # since not hanlding scales yet
        return self.merge_outputs(dets)

    def merge_outputs(self, detections):
        results = []
        for j in range(self.num_classes):
            results.append(
                np.concatenate([detection[j] for detection in detections],
                               axis=0).astype(np.float32))
        # if len(self.scales) > 1 or self.opt.nms:
        #    soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack([results[j][:, 4] for j in range(self.num_classes)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def simple_test(self, img, img_meta, **kwargs):
        with torch.no_grad():
            output = self.extract_feat(img)
            if self.rpn_head:
                output = self.rpn_head(output)[-1]

            dets = ctdet_decode(
                output['hm'].sigmoid_(), output['wh'], reg=output['reg'])

            if self.test_cfg['debug'] >= 2:
                self.debug(self.debugger, img,
                           dets, output)
            # does not test multiple scales yet
            results = self.post_process_test(dets, img_meta[-1])
            return results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with Augmentations

        * Testing with flip augmentation
        * TODO - Multi scale testing
        """
        with torch.no_grad():
            imgs = torch.cat(imgs, 0)
            output = self.extract_feat(imgs)
            if self.rpn_head:
                output = self.rpn_head(output)[-1]

            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']
            # if self.flip_test:
            hm = (hm[0:1] + torch.flip(hm[1:2], [3])) / 2
            wh = (wh[0:1] + torch.flip(wh[1:2], [3])) / 2
            reg = reg[0:1]

            dets = ctdet_decode(hm, wh, reg=reg)
            if self.test_cfg['debug'] >= 2:
                self.debug(self.debugger, imgs,
                           dets, output)
            # does not test multiple scales yet
            results = self.post_process_test(dets, img_metas[-1][-1])
            return results

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= 4
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.test_cfg['img_norm_cfg']['std'] +
                    self.test_cfg['img_norm_cfg']['mean']) * 255).astype(
                        np.uint8)
            pred = debugger.gen_colormap(
                output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > 0.1:
                    debugger.add_coco_bbox(
                        detection[i, k, :4],
                        detection[i, k, -1],
                        detection[i, k, 4],
                        img_id='out_pred_{:.1f}'.format(scale))

    def show_result(self, data, results, img_norm):
        image = data['img'][0].numpy().transpose(1, 2, 0)
        # import pdb; pdb.set_trace()
        self.debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > 0.3:
                    self.debugger.add_coco_bbox(
                        bbox[:4], j - 1, bbox[4], img_id='ctdet')
        self.debugger.show_all_imgs(pause=True)
