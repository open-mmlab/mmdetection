# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, constant_init

from mmdet.models.utils.transformer import inverse_sigmoid
from ..builder import HEADS
from .detr_head import DETRHead


@HEADS.register_module()
class ConditionalDETRHead(DETRHead):

    def __init__(self, *args, **kwargs):
        super(ConditionalDETRHead, self).__init__(*args, **kwargs)

    def init_weights(self):
        """Initialize weights of the Conditional DETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
        constant_init(self.fc_reg, 0., bias=0.)

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
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        # outs_dec, reference = self.transformer(
        #     x, masks, self.query_embedding.weight, pos_embed)
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                       pos_embed)
        hs, reference = outs_dec

        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.bbox_embed(hs[lvl])
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        all_bbox_preds = torch.stack(outputs_coords)
        all_cls_scores = self.fc_cls(hs)
        return all_cls_scores, all_bbox_preds
