# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from torch import Tensor
import math
from mmdet.registry import MODELS
from .atss_head import ATSSHead
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
import torch.nn.functional as F
from mmcv.ops.nms import batched_nms
from mmengine.structures import InstanceData
from mmdet.structures.bbox import scale_boxes
from mmdet.structures import DetDataSample


class Conv3x3Norm(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 groups=1,
                 deformable=False,
                 bn_type=None):
        super(Conv3x3Norm, self).__init__()

        if deformable:
            self.conv = ModulatedDeformConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                              groups=groups)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups)

        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]

        if bn_type == "bn":
            bn_op = nn.BatchNorm2d(out_channels)
        elif bn_type == "sbn":
            bn_op = nn.SyncBatchNorm(out_channels)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=out_channels)
        if bn_type is not None:
            self.bn = bn_op
        else:
            self.bn = None

    def forward(self, input, **kwargs):
        x = self.conv(input, **kwargs)
        if self.bn:
            x = self.bn(x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DYReLU(nn.Module):
    def __init__(self, inp, oup, reduction=4, lambda_a=1.0, K2=True, use_bias=True, use_spatial=False,
                 init_a=[1.0, 0.0], init_b=[0.0, 0.0]):
        super(DYReLU, self).__init__()
        self.oup = oup
        self.lambda_a = lambda_a * 2
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.use_bias = use_bias
        if K2:
            self.exp = 4 if use_bias else 2
        else:
            self.exp = 2 if use_bias else 1
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        if reduction == 4:
            squeeze = inp // reduction
        else:
            squeeze = _make_divisible(inp // reduction, 4)
        # print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
        # print('init_a: {}, init_b: {}'.format(self.init_a, self.init_b))

        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, oup * self.exp),
            h_sigmoid()
        )
        if use_spatial:
            self.spa = nn.Sequential(
                nn.Conv2d(inp, 1, kernel_size=1),
                nn.BatchNorm2d(1),
            )
        else:
            self.spa = None

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)
        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]

            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            if self.use_bias:  # bias but not PL
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                b1 = b1 - 0.5 + self.init_b[0]
                out = x_out * a1 + b1

            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
                out = torch.max(x_out * a1, x_out * a2)

        elif self.exp == 1:
            a1 = y
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            out = x_out * a1

        if self.spa:
            ys = self.spa(x_in).view(b, -1)
            ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
            ys = F.hardtanh(ys, 0, 3, inplace=True) / 3
            out = out * ys

        return out


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


class DyConv(torch.nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 conv_func=nn.Conv2d,
                 use_dyfuse=True,
                 use_dyrelu=False,
                 use_deform=False
                 ):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))

        if use_dyfuse:
            self.AttnConv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.ReLU(inplace=True))
            self.h_sigmoid = h_sigmoid()
        else:
            self.AttnConv = None

        if use_dyrelu:
            self.relu = DYReLU(in_channels, out_channels)
        else:
            self.relu = nn.ReLU()

        if use_deform:
            self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        else:
            self.offset = None

        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.AttnConv is not None:
            for m in self.AttnConv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, inputs):
        visual_feats = inputs["visual"]
        language_dict_features = inputs["lang"]

        next_x = []
        for level, feature in enumerate(visual_feats):

            conv_args = dict()
            if self.offset is not None:
                offset_mask = self.offset(feature)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, 18:, :, :].sigmoid()
                conv_args = dict(offset=offset, mask=mask)

            temp_fea = [self.DyConv[1](feature, **conv_args)]

            if level > 0:
                temp_fea.append(self.DyConv[2](visual_feats[level - 1], **conv_args))
            if level < len(visual_feats) - 1:
                temp_fea.append(F.upsample_bilinear(self.DyConv[0](visual_feats[level + 1], **conv_args),
                                                    size=[feature.size(2), feature.size(3)]))
            mean_fea = torch.mean(torch.stack(temp_fea), dim=0, keepdim=False)

            if self.AttnConv is not None:
                attn_fea = []
                res_fea = []
                for fea in temp_fea:
                    res_fea.append(fea)
                    attn_fea.append(self.AttnConv(fea))

                res_fea = torch.stack(res_fea)
                spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))

                mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)

            next_x.append(mean_fea)

        next_x = [self.relu(item) for item in next_x]

        features_dict = {"visual": next_x,
                         "lang": language_dict_features}

        return features_dict


from mmengine.model import BaseModel


class VLFusionModule(BaseModel):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_base_priors,
                 num_classes,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_base_priors = num_base_priors
        self.num_classes = num_classes
        self._init_layers()

    def _init_layers(self) -> None:
        use_dyrelu = True
        use_dyfuse = True
        use_deform = True
        bn_type = ['gn', 16]
        num_dyhead_blocks = 6
        conv_func = lambda i, o, s: Conv3x3Norm(i, o, s, deformable=use_deform, bn_type=bn_type)
        log_scale = 0.0
        prior_prob = 0.01
        lang_dim = 768

        bias_value = -math.log((1 - prior_prob) / prior_prob)

        dyhead_tower = []
        for i in range(num_dyhead_blocks):
            dyhead_tower.append(
                DyConv(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    conv_func=conv_func,
                    use_dyrelu=(use_dyrelu and self.in_channels == self.feat_channels) if i == 0 else use_dyrelu,
                    use_dyfuse=(use_dyfuse and self.in_channels == self.feat_channels) if i == 0 else use_dyfuse,
                    use_deform=(use_deform and self.in_channels == self.feat_channels) if i == 0 else use_deform,
                )
            )

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))

        self.cls_logits = nn.Conv2d(self.feat_channels, self.num_base_priors * self.num_classes, kernel_size=1)
        self.bbox_pred = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, kernel_size=1)  # num_anchors=1
        self.centerness = nn.Conv2d(self.feat_channels, self.num_base_priors * 1, kernel_size=1)

        self.dot_product_projection_image = nn.Identity()
        # 将语言模型输出进行投影到视觉语义上
        self.dot_product_projection_text = nn.Linear(lang_dim,
                                                     self.num_base_priors * self.feat_channels, bias=True)
        self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)
        # DEBUG
        # self.bias = nn.Parameter(torch.zeros(channels), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(lang_dim), requires_grad=True)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

    def forward(self,
                visual_feats: Tuple[Tensor],
                language_feats: dict):
        logits = []
        bbox_reg = []
        centerness = []

        feat_inputs = {"visual": visual_feats,
                       "lang": language_feats}

        dyhead_tower = self.dyhead_tower(feat_inputs)

        dot_product_logits = []

        USE_DOT_PRODUCT_TOKEN_LOSS = True
        if USE_DOT_PRODUCT_TOKEN_LOSS:
            embedding = language_feats['embedded']
        else:
            embedding = dyhead_tower["lang"]["hidden"]

        # norm
        embedding = F.normalize(embedding, p=2, dim=-1)  # text embeding (1,256,768)

        # 语言特征投影到视觉空间
        dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)  # (1,256,256)
        # print(embedding.sum(), dot_product_proj_tokens.sum())
        dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0  # (1, 256)

        for l, feature in enumerate(visual_feats):
            logits.append(self.cls_logits(dyhead_tower["visual"][l]))  # (1,80,100,136)

            bbox_pred = self.scales[l](self.bbox_pred(dyhead_tower["visual"][l]))
            bbox_reg.append(bbox_pred)

            centerness.append(self.centerness(dyhead_tower["visual"][l]))

            x = dyhead_tower["visual"][l]
            B, C, H, W = x.shape

            # add bias (language)
            # 图像特征作为 query，文本特征作为 key，计算相似度
            dot_product_proj_queries = self.dot_product_projection_image(x)
            dot_product_proj_queries = permute_and_flatten(dot_product_proj_queries, B, -1, C, H, W)  # 1,13600,256

            A = dot_product_proj_queries.shape[1]
            bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)
            # dot_product_proj_tokens 融合后的文本特征 1,13600,256
            dot_product_logit = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1,
                                                                                                          -2)) / self.log_scale.exp()) + bias
            # print(x.sum(), dot_product_logit.sum(), dot_product_proj_queries.sum(), dot_product_proj_tokens.sum(),
            #       self.log_scale)

            dot_product_logit = torch.clamp(dot_product_logit, max=50000)
            dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
            dot_product_logits.append(dot_product_logit)

        return logits, bbox_reg, centerness, dot_product_logits


def convert_grounding_to_od_logits(logits, box_cls, positive_maps, score_agg=None):
    # 这个 scores 维度是 (1,13600,80)，这个 80 是明显不合理的，这个地方应该是当前句子中 token 的个数
    # 假设当前句子一共 3 个命名实体，那么这个维度应该是 (1,13600,3)
    # 虽然结果一样，但是含义就不一样，当某一种图片的实体超过 80 那就会报错了
    assert len(positive_maps) == logits.shape[0]

    scores = torch.zeros(logits.shape[0], logits.shape[1], box_cls.shape[2]).to(logits.device)  # (1,13600,80)
    # 256 -> 80, average for each class
    if positive_maps is not None:
        if all(x == positive_maps[0] for x in positive_maps):
            # only need to compute once
            positive_map = positive_maps[0]
            # score aggregation method
            if score_agg == "MEAN":  # ture
                for label_j in positive_map:  # logits (1,13600,256) 取出对应 token 位置的预测值，然后求均值,将其转换为 80 类的预测值
                    scores[:, :, label_j - 1] = logits[:, :, torch.LongTensor(positive_map[label_j])].mean(-1)
            elif score_agg == "MAX":
                # torch.max() returns (values, indices)
                for label_j in positive_map:
                    scores[:, :, label_j - 1] = logits[:, :, torch.LongTensor(positive_map[label_j])].max(-1)[
                        0]
            elif score_agg == "ONEHOT":
                # one hot
                scores = logits[:, :, :len(positive_map)]
            else:
                raise NotImplementedError
        else:
            for i, positive_map in enumerate(positive_maps):
                if score_agg == "MEAN":  # ture
                    for label_j in positive_map:  # logits (1,13600,256) 取出对应 token 位置的预测值，然后求均值,将其转换为 80 类的预测值
                        scores[i, :, label_j - 1] = logits[i, :, torch.LongTensor(positive_map[label_j])].mean(-1)
                elif score_agg == "MAX":
                    # torch.max() returns (values, indices)
                    for label_j in positive_map:
                        scores[i, :, label_j - 1] = logits[i, :, torch.LongTensor(positive_map[label_j])].max(-1)[
                            0]
                elif score_agg == "ONEHOT":
                    # one hot
                    raise NotImplementedError
                else:
                    raise NotImplementedError

    return scores


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # WORK AROUND: work around unbind using split + squeeze.
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.split(1, dim=1)
    ws = ws.squeeze(1)
    hs = hs.squeeze(1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


class ATSSPostProcessor(torch.nn.Module):
    def __init__(
            self,
            pre_nms_thresh=0.05,
            pre_nms_top_n=1000,
            nms_thresh=0.6,
            fpn_post_nms_top_n=100,
            min_size=0,
            box_coder=None,
            score_agg='MEAN',
            cfg=None
    ):
        super(ATSSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.box_coder = box_coder
        self.score_agg = score_agg
        self.cfg = cfg

    def select_over_all_levels(self, bboxes, scores, labels, metainfo):

        num_images = len(bboxes)
        results = []
        for i in range(num_images):
            pred_instance = InstanceData()
            if bboxes[i].numel() == 0:
                pred_instance.bboxes = bboxes[i]
                pred_instance.scores = scores[i]
                pred_instance.labels = labels[i]
                results.append(pred_instance)
            else:
                det_bboxes, keep_idxs = batched_nms(bboxes[i], scores[i], labels[i], nms_cfg=self.cfg.nms)

                pred_instance.bboxes = det_bboxes[:, :4]
                pred_instance.scores = det_bboxes[:, -1]
                pred_instance.labels = labels[i][keep_idxs]

                # multiclass nms
                # result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
                number_of_detections = len(pred_instance)

                if number_of_detections == 0:
                    results.append(pred_instance)
                    continue

                scale_factor = [1 / s for s in metainfo[i]['scale_factor']]
                pred_instance.bboxes = scale_boxes(pred_instance.bboxes, scale_factor)

                # Limit to max_per_image detections **over all classes**
                if number_of_detections > self.fpn_post_nms_top_n > 0:
                    cls_scores = pred_instance.scores
                    image_thresh, _ = torch.kthvalue(
                        cls_scores.cpu().float(),
                        number_of_detections - self.fpn_post_nms_top_n + 1
                    )
                    keep = cls_scores >= image_thresh.item()
                    keep = torch.nonzero(keep).squeeze(1)
                    pred_instance = pred_instance[keep]
                results.append(pred_instance)
        return results

    def forward_for_single_feature_map(self,
                                       box_regression,
                                       centerness,
                                       anchors,
                                       box_cls=None,
                                       dot_product_logits=None,
                                       positive_maps=None,
                                       metainfo=None
                                       ):

        N, _, H, W = box_regression.shape

        A = box_regression.size(1) // 4

        if box_cls is not None:
            C = box_cls.size(1) // A

        # put in the same format as anchors
        if box_cls is not None:
            # print('Classification.')
            box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
            box_cls = box_cls.sigmoid()  # (1,13600,80)

        dot_product_logits = dot_product_logits.sigmoid()  # (1,136000,80)
        scores = convert_grounding_to_od_logits(logits=dot_product_logits, box_cls=box_cls,
                                                positive_maps=positive_maps,
                                                score_agg=self.score_agg)
        box_cls = scores

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        centerness = permute_and_flatten(centerness, N, A, 1, H, W)
        centerness = centerness.reshape(N, -1).sigmoid()

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []

        for per_box_cls, per_box_regression, per_pre_nms_top_n, per_candidate_inds, img_meta in zip(box_cls,
                                                                                                    box_regression,
                                                                                                    pre_nms_top_n,
                                                                                                    candidate_inds,
                                                                                                    metainfo):

            img_shape = img_meta['img_shape']

            per_box_cls = per_box_cls[per_candidate_inds]

            per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
            # (13600, 80)  -> (n, 2)
            per_candidate_nonzeros = per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            # print(per_box_regression.sum(), anchors.sum())
            bboxes = self.box_coder.decode(
                anchors[per_box_loc, :].view(-1, 4),
                per_box_regression[per_box_loc, :].view(-1, 4),
                max_shape=img_shape
            )
            scores = torch.sqrt(per_box_cls)

            if self.min_size >= 0 and per_pre_nms_top_n > 0:
                w = bboxes[:, 2] - bboxes[:, 0]
                h = bboxes[:, 3] - bboxes[:, 1]
                valid_mask = (w > self.min_size) & (h > self.min_size)
                if not valid_mask.all():
                    bboxes = bboxes[valid_mask]
                    scores = scores[valid_mask]
                    per_class = per_class[valid_mask]

            # boxlist = BoxList(detections, anchors.size, mode="xyxy")
            # boxlist.add_field("labels", per_class)
            # boxlist.add_field("scores", torch.sqrt(per_box_cls))
            # boxlist = boxlist.clip_to_image(remove_empty=False)
            # boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(dict(bboxes=bboxes, scores=scores, per_class=per_class))
        return results

    def forward(self,
                box_regression,
                centerness,
                anchors,
                box_cls=None,
                dot_product_logits=None,
                positive_maps=None,
                metainfo=None
                ):
        sampled_boxes = []
        for idx, (b, c, a) in enumerate(zip(box_regression, centerness, anchors)):
            o = box_cls[idx]
            d = dot_product_logits[idx]
            sampled_boxes.append(
                self.forward_for_single_feature_map(b, c, a, o, d, positive_maps, metainfo)
            )
        boxlists = list(zip(*sampled_boxes))

        bboxes = []
        scores = []
        labels = []
        for boxlist in boxlists:
            bboxes.append(torch.cat([box['bboxes'] for box in boxlist]))
            scores.append(torch.cat([box['scores'] for box in boxlist]))
            labels.append(torch.cat([box['per_class'] for box in boxlist]))

        # boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(bboxes, scores, labels, metainfo)
        return boxlists


@MODELS.register_module()
class ATSSVLFusionHead(ATSSHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = VLFusionModule(in_channels=self.in_channels,
                                   feat_channels=self.feat_channels,
                                   num_base_priors=self.num_base_priors,
                                   num_classes=self.num_classes)
        self.postprocess = ATSSPostProcessor(box_coder=self.bbox_coder, cfg=self.test_cfg)

    def _init_layers(self) -> None:
        pass

    def predict(self,
                visual_feats: Tuple[Tensor],
                language_feats: dict,
                batch_data_samples,
                rescale: bool = True):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_token_positive_maps = [
            data_samples.token_positive_map for data_samples in batch_data_samples
        ]

        featmap_sizes = [visual_feats[i].shape[-2:] for i in range(len(visual_feats))]

        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=visual_feats[0].dtype,
            device=visual_feats[0].device)

        box_cls, box_regression, centerness, dot_product_logits = self.head(
            visual_feats,
            language_feats
        )
        results = self.postprocess(box_regression,
                                   centerness,
                                   mlvl_priors,
                                   box_cls,
                                   dot_product_logits,
                                   positive_maps=batch_token_positive_maps,
                                   metainfo=batch_img_metas)

        return results
