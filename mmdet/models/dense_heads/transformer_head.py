import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, xavier_init
from mmcv.runner import force_fp32

from mmdet.core import bbox_cxcywh_to_xyxy, build_sampler, multi_apply
from mmdet.models.utils import FFN, build_position_encoding, build_transformer
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class TransformerHead(AnchorFreeHead):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (dict, optional): Config for transformer.
        position_encoding (dict, optional): Config for position encoding.
        loss_cls (dict, optional): Config of the classification loss.
            Default `CrossEntropyLoss`.
        loss_bbox (dict, optional): Config of the regression loss.
            Default `L1Loss`.
        loss_giou (dict, optional): Config of the regression giou loss.
            Default `GIoULoss`.
        tran_cfg (dict, optional): Training config of transformer head.
        test_cfg (dict, optional): Testing config of transformer head.

    Example:
        >>> import torch
        >>> self = TransformerHead(80, 2048)
        >>> x = torch.rand(1, 2048, 32, 32)
        >>> mask = torch.ones(1, 32, 32).to(x.dtype)
        >>> mask[:, :16, :15] = 0
        >>> all_cls_scores, all_bbox_preds = self(x, mask)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_fcs=2,
                 transformer=dict(
                     type='Transformer',
                     embed_dims=256,
                     num_heads=8,
                     num_encoder_layers=6,
                     num_decoder_layers=6,
                     feedforward_channels=2048,
                     dropout=0.1,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN'),
                     num_fcs=2,
                     pre_norm=False,
                     return_intermediate_dec=True),
                 position_encoding=dict(
                     type='SinePositionEmbedding',
                     num_feats=128,
                     normalize=True),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_giou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianMatcher',
                         cls_weight=1.,
                         bbox_weight=5.,
                         giou_weight=2.),
                     pos_weight=-1),
                 test_cfg=dict(max_per_img=100),
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__()
        use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        assert not use_sigmoid_cls, 'setting use_sigmoid_cls as True is ' \
            'not supported in DETR, since background is needed for the ' \
            'matching process.'
        assert 'embed_dims' in transformer and 'num_feats' in position_encoding
        num_feats = position_encoding['num_feats']
        embed_dims = transformer['embed_dims']
        assert num_feats * 2 == embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {embed_dims}' \
            f' and {num_feats}.'
        assert test_cfg is not None and 'max_per_img' in test_cfg

        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None:
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['bbox_weight'], \
                'The regression L1 weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_giou['loss_weight'] == assigner['giou_weight'], \
                'The regression giou weight for loss and matcher should be' \
                'exactly the same.'
            # TODO self.assigner = build_assigner(assigner)
            self.assigner = None
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes + 1
        self.in_channels = in_channels
        self.num_fcs = num_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_sigmoid_cls = use_sigmoid_cls
        self.embed_dims = embed_dims
        self.num_query = test_cfg['max_per_img']
        self.background_label = num_classes
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_giou = build_loss(loss_giou)
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.position_encoding = build_position_encoding(position_encoding)
        self.transformer = build_transformer(transformer)
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_fcs,
            self.act_cfg,
            add_residual=False)
        self.fc_reg = Linear(self.embed_dims, 4)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def init_weights(self, distribution='uniform'):
        """Initialize weights of the transformer head."""
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution=distribution)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.
        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single, feats, img_metas_list)

    def forward_single(self, x, img_metas):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['input_img_shape']
        masks = torch.ones((batch_size, input_img_h, input_img_w)).to(x.device)
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0
        masks = masks.to(x.dtype)

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.position_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                       pos_embed)
        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds

    @force_fp32(apply_to=('all_cls_scores', 'all_bbox_preds'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Box score logits for all the decoder
                layers. Shape [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (list[Tensor]): Sigmoid outputs with normalized
                coordinate format (cx, cy, w, h). Shape [nb_dec, bs,
                num_query, 4].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Defalut False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the ouputs from the last decoder layer is used.
        cls_scores = all_cls_scores[-1][-1]
        bbox_preds = all_bbox_preds[-1][-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        # exclude background
        scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)
        return det_bboxes, det_labels

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """Compute regression, classification and centerss targets."""
        pass

    @force_fp32(apply_to=('all_cls_scores', 'all_bbox_preds'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head."""
        pass
