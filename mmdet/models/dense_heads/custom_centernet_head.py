import torch
import torch.nn as nn
import math
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from mmdet.models.utils import _transpose
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin

from torch.nn import functional as F
from .centernet_head import CenterNetHead
from mmdet.models.utils.centernet2_utils import Instances, Boxes, ml_nms

INF = 100000000

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            # "BN": BatchNorm2d,
            # # Fixed in https://github.com/pytorch/pytorch/pull/36382
            # "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            # "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            # "naiveSyncBN": NaiveSyncBatchNorm,
        }[norm]
    return norm(out_channels)


@HEADS.register_module()
class CustomCenterNetHead(BaseDenseHead, BBoxTestMixin):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                #  only_proposal,
                #  with_agn_hm,
                #  more_pos,
                #  strides,
                #  sizes_of_interest,
                 loss_center_heatmap=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(CustomCenterNetHead, self).__init__(init_cfg)
        self.num_classes = num_classes

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        self.out_kernel = 3
        norm = "GN"
        self.only_proposal = True
        self.with_agn_hm = True
        self.more_pos = False
        self.strides = [8, 16, 32, 64, 128]
        self.sizes_of_interest = [[0,80],[64,160],[128,320],[256,640],[512,10000000]]
        self.delta = (1 - 0.8) / (1 + 0.8)
        self.min_radius = 4
        self.center_nms = False
        self.pre_nms_topk_train = 4000
        self.pre_nms_topk_test = 1000
        self.nms_thresh_train = 0.9
        self.nms_thresh_test = 0.9
        self.post_nms_topk_train = 2000
        self.post_nms_topk_test = 256
        self.more_pos_topk = 9
        self.score_thresh = 0.0001
        self.not_nms = False
        

        ##########  initialize the   1.<cls_tower>   2.<bbox_tower>   3.<share_tower>[no]  4.<bbox_pred>  5.<agn_hm>  6.<cls_logits>[no]

        ######################################################## origin #########################
        # self.heatmap_head = self._build_head(in_channel, feat_channel,
        #                                      num_classes)
        # self.wh_head = self._build_head(in_channel, feat_channel, 2)
        # self.offset_head = self._build_head(in_channel, feat_channel, 2)


        ########################################################## lq #########################
        # head_configs = {"cls": (cfg.MODEL.CENTERNET.NUM_CLS_CONVS \
        #                         if not self.only_proposal else 0,
        #                         cfg.MODEL.CENTERNET.USE_DEFORMABLE),
        #                 "bbox": (cfg.MODEL.CENTERNET.NUM_BOX_CONVS,
        #                          cfg.MODEL.CENTERNET.USE_DEFORMABLE),
        #                 "share": (cfg.MODEL.CENTERNET.NUM_SHARE_CONVS,
        #                           cfg.MODEL.CENTERNET.USE_DEFORMABLE)}

        head_configs = {"cls": (4,False),
                        "bbox": (4,False),
                        "share": (0,False)}

        #############  centernet2 , channels from ["p3", "p4", "p5", "p6", "p7"]'s channels, config
        # in_channels = [s.channels for s in input_shape]
        # assert len(set(in_channels)) == 1, \
        #     "Each level must have the same channel!"
        # in_channels = in_channels[0]

        channels = {
            'cls': in_channel,
            'bbox': in_channel,
            'share': in_channel,
        }

        ##### initialize the     1.<cls_tower>    2.<bbox_tower>     3.<share_tower>
        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            channel = channels[head]
            for i in range(num_convs):
                conv_func = nn.Conv2d
                tower.append(conv_func(
                        in_channel if i == 0 else channel,
                        channel,
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                ))
                if norm == 'GN' and channel % 32 != 0:
                    tower.append(nn.GroupNorm(25, channel))
                elif norm != '':
                    # print("please add get_norm function")
                    tower.append(get_norm(norm, channel))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        ### initialize the    <bbox_pred>
        self.bbox_pred = nn.Conv2d(
            in_channel, 4, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )

        ### initialize the     <scales>
        self.scales = Scale(init_value=1.0)
        # self.scales = nn.ModuleList(
        #     [Scale(init_value=1.0) for _ in input_shape])

        ### initialize the     <agn_hm>
        self.agn_hm = nn.Conv2d(
            in_channel, 1, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )

        ### initialize the <cls_logits>, config assigns it to false !
        if not self.only_proposal:
            cls_kernel_size = self.out_kernel
            self.cls_logits = nn.Conv2d(
                in_channel, self.num_classes,
                kernel_size=cls_kernel_size,
                stride=1,
                padding=cls_kernel_size // 2,
            )


    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer


    def init_weights(self):
        """Initialize weights of the head."""
        ###################### origin ######################
        # bias_init = bias_init_with_prob(0.1)
        # self.heatmap_head[-1].bias.data.fill_(bias_init)
        # for head in [self.wh_head, self.offset_head]:
        #     for m in head.modules():
        #         if isinstance(m, nn.Conv2d):
        #             normal_init(m, std=0.001)

        ########################  lq #####################################

        ### initialize the    1.<cls_tower>   2.<bbox_tower>    3.<share_tower>   4.<bbox_pred>
        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower,
            self.bbox_pred,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        torch.nn.init.constant_(self.bbox_pred.bias, 8.)

        ### initialize the    <agn_hm>
        #prior_prob = cfg.MODEL.CENTERNET.PRIOR_PROB --> 0.01 in config
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        torch.nn.init.constant_(self.agn_hm.bias, bias_value)
        torch.nn.init.normal_(self.agn_hm.weight, std=0.01)

        ##############  if <cls_logits>
        if not self.only_proposal:
            torch.nn.init.constant_(self.cls_logits.bias, bias_value)
            torch.nn.init.normal_(self.cls_logits.weight, std=0.01)


    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        # clss_per_level, reg_pred_per_level, agn_hm_pred_per_level = \
        #     multi_apply(self.forward_single, feats)
        
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            cls(Tensor): cls predicts, the channels number is class number: 80  # not used
            bbox_reg(Tensor): reg predicts, the channels number is 4
            agn_hms (Tensor): center predict heatmaps, the channels number is 1

        """
        feat = self.share_tower(feat)
        cls_tower = self.cls_tower(feat)
        bbox_tower = self.bbox_tower(feat)
        if not self.only_proposal:
            clss = self.cls_logits(cls_tower)
        else:
            clss = None

        if self.with_agn_hm:
            agn_hms = self.agn_hm(bbox_tower)
        else:
            agn_hms = None
        reg = self.bbox_pred(bbox_tower)
        reg = self.scales(reg)
        return clss, reg, agn_hms


    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]

        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
                                                     center_heatmap_pred.shape,
                                                     img_metas[0]['pad_shape'])

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        # Since the channel of wh_target and offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
        loss_wh = self.loss_wh(
            wh_pred,
            wh_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset)

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight)
        return target_result, avg_factor
    '''
    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   img_metas,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
        border_pixs = [img_meta['border'] for img_meta in img_metas]

        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            wh_preds[0],
            offset_preds[0],
            img_metas[0]['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        batch_border = batch_det_bboxes.new_tensor(
            border_pixs)[:, [2, 0, 2, 0]].unsqueeze(1)
        batch_det_bboxes[..., :4] -= batch_border

        if rescale:
            batch_det_bboxes[..., :4] /= batch_det_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        if with_nms:
            det_results = []
            for (det_bboxes, det_labels) in zip(batch_det_bboxes,
                                                batch_labels):
                det_bbox, det_label = self._bboxes_nms(det_bboxes, det_labels,
                                                       self.test_cfg)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
            ]
        return det_results
    '''
    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels

        out_bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1], labels,
                                       cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels


#################################
#################################
    def get_bboxes(self, *feats, img_metas, gt_bboxes, gt_labels):

        grids = self.compute_grids(feats)
        # grids = multi_apply(self.compute_grids, feats, self.strides)
        shapes_per_level = grids[0].new_tensor(
                    [(x.shape[2], x.shape[3]) for x in reg_pred_per_level])

        if self.training:
            
            if self.more_pos:
                # add more pixels as positive if \
                #   1. they are within the center3x3 region of an object
                #   2. their regression losses are small (<self.more_pos_thresh)
                pos_inds, labels = self._add_more_pos(
                    reg_pred, gt_bboxes, gt_labels, shapes_per_level)
            
            proposals = None
            image_sizes = []
            for i in range(len(img_meta)):
                image_sizes.append(img_meta[i]['ori_shape'][:2])
            if self.only_proposal:
                agn_hm_pred_per_level = [x.sigmoid() for x in agn_hm_pred_per_level]
                proposals = self.predict_instances(
                    grids, agn_hm_pred_per_level, reg_pred_per_level, 
                    image_sizes, [None for _ in agn_hm_pred_per_level])
            elif self.as_proposal: # category specific bbox as agnostic proposals
                clss_per_level = [x.sigmoid() for x in clss_per_level]
                proposals = self.predict_instances(
                    grids, clss_per_level, reg_pred_per_level, 
                    image_sizes, agn_hm_pred_per_level)
            if self.only_proposal or self.as_proposal:
                for p in range(len(proposals)):
                    proposals[p].proposal_boxes = proposals[p].get('pred_boxes')
                    proposals[p].objectness_logits = proposals[p].get('scores')
                    proposals[p].remove('pred_boxes')
                    proposals[p].remove('scores')
                    proposals[p].remove('pred_classes')

            # losses = self.losses(
            #     pos_inds, labels, reg_targets, flattened_hms,
            #     logits_pred, reg_pred, agn_hm_pred)

            # if self.debug:
            #     debug_train(
            #         [self.denormalizer(x) for x in images], 
            #         gt_instances, flattened_hms, reg_targets, 
            #         labels, pos_inds, shapes_per_level, grids, self.strides)
            return proposals



    def compute_grids(self, features):
        
        grids = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            shifts_x = torch.arange(
                0, w * self.strides[level], 
                step=self.strides[level],
                dtype=torch.float32, device=feature.device)
            shifts_y = torch.arange(
                0, h * self.strides[level], 
                step=self.strides[level],
                dtype=torch.float32, device=feature.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            grids_per_level = torch.stack((shift_x, shift_y), dim=1) + \
                self.strides[level] // 2
            grids.append(grids_per_level)
        return grids

    def _get_label_inds(self, gt_bboxes, gt_labels, shapes_per_level):
        '''
        Inputs:
            gt_instances: [n_i], sum n_i = N
            shapes_per_level: L x 2 [(h_l, w_l)]_L
        Returns:
            pos_inds: N'
            labels: N'
        '''
        pos_inds = []
        labels = []
        L = len(self.strides)
        B = len(gt_labels)
        shapes_per_level = shapes_per_level.long()
        loc_per_level = (shapes_per_level[:, 0] * shapes_per_level[:, 1]).long() # L
        level_bases = []
        s = 0
        for l in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[l]
        level_bases = shapes_per_level.new_tensor(level_bases).long() # L
        strides_default = shapes_per_level.new_tensor(self.strides).float() # L
        for im_i in range(B):
            # targets_per_im = gt_instances[im_i]
            bboxes = gt_bboxes[im_i]   #.tensor # n x 4
            n = bboxes.shape[0]
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2) # n x 2
            centers = centers.view(n, 1, 2).expand(n, L, 2)
            strides = strides_default.view(1, L, 1).expand(n, L, 2)
            centers_inds = (centers / strides).long() # n x L x 2
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            pos_ind = level_bases.view(1, L).expand(n, L) + \
                       im_i * loc_per_level.view(1, L).expand(n, L) + \
                       centers_inds[:, :, 1] * Ws + \
                       centers_inds[:, :, 0] # n x L
            is_cared_in_the_level = self.assign_fpn_level(bboxes)
            pos_ind = pos_ind[is_cared_in_the_level].view(-1)
            label = gt_labels[im_i].view(
                n, 1).expand(n, L)[is_cared_in_the_level].view(-1)

            pos_inds.append(pos_ind) # n'
            labels.append(label) # n'
        pos_inds = torch.cat(pos_inds, dim=0).long()
        labels = torch.cat(labels, dim=0)
        return pos_inds, labels # N, N
    
    def _get_reg_targets(self, reg_targets, dist, mask, area):
        '''
          reg_targets (M x N x 4): long tensor
          dist (M x N)
          is_*: M x N
        '''
        dist[mask == 0] = INF * 1.0
        min_dist, min_inds = dist.min(dim=1) # M
        reg_targets_per_im = reg_targets[
            range(len(reg_targets)), min_inds] # M x N x 4 --> M x 4
        reg_targets_per_im[min_dist == INF] = - INF
        return reg_targets_per_im


    def _create_heatmaps_from_dist(self, dist, labels, channels):
        '''
        dist: M x N
        labels: N
        return:
          heatmaps: M x C
        '''
        heatmaps = dist.new_zeros((dist.shape[0], channels))
        for c in range(channels):
            inds = (labels == c) # N
            if inds.int().sum() == 0:
                continue
            heatmaps[:, c] = torch.exp(-dist[:, inds].min(dim=1)[0])
            zeros = heatmaps[:, c] < 1e-4
            heatmaps[zeros, c] = 0
        return heatmaps

    def inference(self, images, clss_per_level, reg_pred_per_level, 
        agn_hm_pred_per_level, grids):
        logits_pred = [x.sigmoid() if x is not None else None \
            for x in clss_per_level]
        agn_hm_pred_per_level = [x.sigmoid() if x is not None else None \
            for x in agn_hm_pred_per_level]

        if self.only_proposal:
            proposals = self.predict_instances(
                grids, agn_hm_pred_per_level, reg_pred_per_level, 
                images.image_sizes, [None for _ in agn_hm_pred_per_level])
        else:
            proposals = self.predict_instances(
                grids, logits_pred, reg_pred_per_level, 
                images.image_sizes, agn_hm_pred_per_level)
        if self.as_proposal or self.only_proposal:
            for p in range(len(proposals)):
                proposals[p].proposal_boxes = proposals[p].get('pred_boxes')
                proposals[p].objectness_logits = proposals[p].get('scores')
                proposals[p].remove('pred_boxes')

        # if self.debug:
        #     debug_test(
        #         [self.denormalizer(x) for x in images], 
        #         logits_pred, reg_pred_per_level, 
        #         agn_hm_pred_per_level, preds=proposals,
        #         vis_thresh=self.vis_thresh, 
        #         debug_show_name=False)
        return proposals, {}


    def predict_instances(
        self, grids, logits_pred, reg_pred, image_sizes, agn_hm_pred, 
        is_proposal=False):
        sampled_boxes = []
        for l in range(len(grids)):
            sampled_boxes.append(self.predict_single_level(
                grids[l], logits_pred[l], reg_pred[l] * self.strides[l],
                image_sizes, agn_hm_pred[l], l, is_proposal=is_proposal))
        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.nms_and_topK(
            boxlists, nms=not self.not_nms)
        return boxlists

    def predict_single_level(
        self, grids, heatmap, reg_pred, image_sizes, agn_hm, level, 
        is_proposal=False):
        N, C, H, W = heatmap.shape
        # put in the same format as grids
        if self.center_nms:
            heatmap_nms = nn.functional.max_pool2d(
                heatmap, (3, 3), stride=1, padding=1)
            heatmap = heatmap * (heatmap_nms == heatmap).float()
        heatmap = heatmap.permute(0, 2, 3, 1) # N x H x W x C
        heatmap = heatmap.reshape(N, -1, C) # N x HW x C
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1) # N x H x W x 4 
        box_regression = box_regression.reshape(N, -1, 4)

        candidate_inds = heatmap > self.score_thresh # 0.05
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1) # N
        pre_nms_topk = self.pre_nms_topk_train if self.training else self.pre_nms_topk_test
        pre_nms_top_n = pre_nms_top_n.clamp(max=pre_nms_topk) # N

        if agn_hm is not None:
            agn_hm = agn_hm.view(N, 1, H, W).permute(0, 2, 3, 1)
            agn_hm = agn_hm.reshape(N, -1)
            heatmap = heatmap * agn_hm[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = heatmap[i] # HW x C
            per_candidate_inds = candidate_inds[i] # n
            per_box_cls = per_box_cls[per_candidate_inds] # n

            per_candidate_nonzeros = per_candidate_inds.nonzero() # n
            per_box_loc = per_candidate_nonzeros[:, 0] # n
            per_class = per_candidate_nonzeros[:, 1] # n

            per_box_regression = box_regression[i] # HW x 4
            per_box_regression = per_box_regression[per_box_loc] # n x 4
            per_grids = grids[per_box_loc] # n x 2

            per_pre_nms_top_n = pre_nms_top_n[i] # 1

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_grids = per_grids[top_k_indices]
            
            detections = torch.stack([
                per_grids[:, 0] - per_box_regression[:, 0],
                per_grids[:, 1] - per_box_regression[:, 1],
                per_grids[:, 0] + per_box_regression[:, 2],
                per_grids[:, 1] + per_box_regression[:, 3],
            ], dim=1) # n x 4

            # avoid invalid boxes in RoI heads
            detections[:, 2] = torch.max(detections[:, 2], detections[:, 0] + 0.01)
            detections[:, 3] = torch.max(detections[:, 3], detections[:, 1] + 0.01)
            boxlist = Instances(image_sizes[i])
            boxlist.scores = torch.sqrt(per_box_cls) \
                if self.with_agn_hm else per_box_cls # n
            # import pdb; pdb.set_trace()
            boxlist.pred_boxes = Boxes(detections)
            boxlist.pred_classes = per_class
            results.append(boxlist)
        return results


    def nms_and_topK(self, boxlists, nms=True):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            nms_thresh = self.nms_thresh_train if self.training else \
                self.nms_thresh_test
            result = ml_nms(boxlists[i], nms_thresh) if nms else boxlists[i]
            # if self.debug:
            #     print('#proposals before nms', len(boxlists[i]))
            #     print('#proposals after nms', len(result))
            num_dets = len(result)
            post_nms_topk = self.post_nms_topk_train if self.training else \
                self.post_nms_topk_test
            if num_dets > post_nms_topk:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    num_dets - post_nms_topk + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            # if self.debug:
            #     print('#proposals after filter', len(result))
            results.append(result)
        return results


    def _add_more_pos(self, reg_pred, gt_instances, shapes_per_level):
        labels, level_masks, c33_inds, c33_masks, c33_regs = \
            self._get_c33_inds(gt_instances, shapes_per_level)
        N, L, K = labels.shape[0], len(self.strides), 9
        c33_inds[c33_masks == 0] = 0
        reg_pred_c33 = reg_pred[c33_inds].detach() # N x L x K
        invalid_reg = c33_masks == 0
        c33_regs_expand = c33_regs.view(N * L * K, 4).clamp(min=0)
        if N > 0:
            with torch.no_grad():
                c33_reg_loss = self.iou_loss(
                    reg_pred_c33.view(N * L * K, 4), 
                    c33_regs_expand, None,
                    reduction='none').view(N, L, K).detach() # N x L x K
        else:
            c33_reg_loss = reg_pred_c33.new_zeros((N, L, K)).detach()
        c33_reg_loss[invalid_reg] = INF # N x L x K
        c33_reg_loss.view(N * L, K)[level_masks.view(N * L), 4] = 0 # real center
        c33_reg_loss = c33_reg_loss.view(N, L * K)
        if N == 0:
            loss_thresh = c33_reg_loss.new_ones((N)).float()
        else:
            loss_thresh = torch.kthvalue(
                c33_reg_loss, self.more_pos_topk, dim=1)[0] # N
        loss_thresh[loss_thresh > self.more_pos_thresh] = self.more_pos_thresh # N
        new_pos = c33_reg_loss.view(N, L, K) < \
            loss_thresh.view(N, 1, 1).expand(N, L, K)
        pos_inds = c33_inds[new_pos].view(-1) # P
        labels = labels.view(N, 1, 1).expand(N, L, K)[new_pos].view(-1)
        return pos_inds, labels
        
    
    def _get_c33_inds(self, gt_instances, shapes_per_level):
        '''
        TODO (Xingyi): The current implementation is ugly. Refactor.
        Get the center (and the 3x3 region near center) locations of each objects
        Inputs:
            gt_instances: [n_i], sum n_i = N
            shapes_per_level: L x 2 [(h_l, w_l)]_L
        '''
        labels = []
        level_masks = []
        c33_inds = []
        c33_masks = []
        c33_regs = []
        L = len(self.strides)
        B = len(gt_instances)
        shapes_per_level = shapes_per_level.long()
        loc_per_level = (shapes_per_level[:, 0] * shapes_per_level[:, 1]).long() # L
        level_bases = []
        s = 0
        for l in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[l]
        level_bases = shapes_per_level.new_tensor(level_bases).long() # L
        strides_default = shapes_per_level.new_tensor(self.strides).float() # L
        K = 9
        dx = shapes_per_level.new_tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1]).long()
        dy = shapes_per_level.new_tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1]).long()
        for im_i in range(B):
            targets_per_im = gt_instances[im_i]
            bboxes = targets_per_im.gt_boxes.tensor # n x 4
            n = bboxes.shape[0]
            if n == 0:
                continue
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2) # n x 2
            centers = centers.view(n, 1, 2).expand(n, L, 2)

            strides = strides_default.view(1, L, 1).expand(n, L, 2) # 
            centers_inds = (centers / strides).long() # n x L x 2
            center_grids = centers_inds * strides + strides // 2# n x L x 2
            l = center_grids[:, :, 0] - bboxes[:, 0].view(n, 1).expand(n, L)
            t = center_grids[:, :, 1] - bboxes[:, 1].view(n, 1).expand(n, L)
            r = bboxes[:, 2].view(n, 1).expand(n, L) - center_grids[:, :, 0]
            b = bboxes[:, 3].view(n, 1).expand(n, L) - center_grids[:, :, 1] # n x L
            reg = torch.stack([l, t, r, b], dim=2) # n x L x 4
            reg = reg / strides_default.view(1, L, 1).expand(n, L, 4).float()
            
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            Hs = shapes_per_level[:, 0].view(1, L).expand(n, L)
            expand_Ws = Ws.view(n, L, 1).expand(n, L, K)
            expand_Hs = Hs.view(n, L, 1).expand(n, L, K)
            label = targets_per_im.gt_classes.view(n).clone()
            mask = reg.min(dim=2)[0] >= 0 # n x L
            mask = mask & self.assign_fpn_level(bboxes)
            labels.append(label) # n
            level_masks.append(mask) # n x L

            Dy = dy.view(1, 1, K).expand(n, L, K)
            Dx = dx.view(1, 1, K).expand(n, L, K)
            c33_ind = level_bases.view(1, L, 1).expand(n, L, K) + \
                       im_i * loc_per_level.view(1, L, 1).expand(n, L, K) + \
                       (centers_inds[:, :, 1:2].expand(n, L, K) + Dy) * expand_Ws + \
                       (centers_inds[:, :, 0:1].expand(n, L, K) + Dx) # n x L x K
            
            c33_mask = \
                ((centers_inds[:, :, 1:2].expand(n, L, K) + dy) < expand_Hs) & \
                ((centers_inds[:, :, 1:2].expand(n, L, K) + dy) >= 0) & \
                ((centers_inds[:, :, 0:1].expand(n, L, K) + dx) < expand_Ws) & \
                ((centers_inds[:, :, 0:1].expand(n, L, K) + dx) >= 0)
            # TODO (Xingyi): think about better way to implement this
            # Currently it hard codes the 3x3 region
            c33_reg = reg.view(n, L, 1, 4).expand(n, L, K, 4).clone()
            c33_reg[:, :, [0, 3, 6], 0] -= 1
            c33_reg[:, :, [0, 3, 6], 2] += 1
            c33_reg[:, :, [2, 5, 8], 0] += 1
            c33_reg[:, :, [2, 5, 8], 2] -= 1
            c33_reg[:, :, [0, 1, 2], 1] -= 1
            c33_reg[:, :, [0, 1, 2], 3] += 1
            c33_reg[:, :, [6, 7, 8], 1] += 1
            c33_reg[:, :, [6, 7, 8], 3] -= 1
            c33_mask = c33_mask & (c33_reg.min(dim=3)[0] >= 0) # n x L x K
            c33_inds.append(c33_ind)
            c33_masks.append(c33_mask)
            c33_regs.append(c33_reg)
        
        if len(level_masks) > 0:
            labels = torch.cat(labels, dim=0)
            level_masks = torch.cat(level_masks, dim=0)
            c33_inds = torch.cat(c33_inds, dim=0).long()
            c33_regs = torch.cat(c33_regs, dim=0)
            c33_masks = torch.cat(c33_masks, dim=0)
        else:
            labels = shapes_per_level.new_zeros((0)).long()
            level_masks = shapes_per_level.new_zeros((0, L)).bool()
            c33_inds = shapes_per_level.new_zeros((0, L, K)).long()
            c33_regs = shapes_per_level.new_zeros((0, L, K, 4)).float()
            c33_masks = shapes_per_level.new_zeros((0, L, K)).bool()
        return labels, level_masks, c33_inds, c33_masks, c33_regs # N x L, N x L x K

# if __name__ == "__main__":
#     centernet2_test = CustomCenterNetHead(in_channel=50, feat_channel=25)
#     centernet2_test.init_weights()
#     print(centernet2_test)
#     feature = torch.randn((16,120,120,16))
#     centernet2_test([feature, feature, feature])