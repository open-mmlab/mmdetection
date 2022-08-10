# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, bias_init_with_prob, constant_init, uniform_init
from mmcv.runner import ModuleList

from mmdet.models.utils.transformer import inverse_sigmoid, MLP
from ..builder import HEADS
from .detr_head import DETRHead


@HEADS.register_module()
class DABDETRHead(DETRHead):
    """Implements the DAB-DETR transformer head.

    See `paper: DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2201.12329>`_ for details.

    Args:
    """

    def __init__(self,
                 *args,
                 iter_update=True,
                 random_refpoints_xy=False,
                 **kwargs):
        self.iter_update = iter_update
        self.random_refpoints_xy = random_refpoints_xy
        super(DABDETRHead, self).__init__(*args, **kwargs)

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.query_dim = self.transformer.query_dim
        self.bbox_embed_diff_each_layer = self.transformer.bbox_embed_diff_each_layer
        self.nb_dec = self.transformer.nb_dec
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        if self.bbox_embed_diff_each_layer:
            self.fc_reg = ModuleList([MLP(self.embed_dims, self.embed_dims, 4, 3)
                                      for _ in range(self.nb_dec)])
        else:
            self.fc_reg = MLP(self.embed_dims, self.embed_dims, 4, 3)
        # TODO: ================== fix build_optimizer error ==================
        # if self.iter_update:
        #     self.transformer.decoder.fc_reg = self.fc_reg
        # TODO: ================== fix build_optimizer error ==================
        self.query_embedding = nn.Embedding(self.num_query, self.query_dim)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        # focal loss initialization
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
        if self.bbox_embed_diff_each_layer:
            for reg in self.fc_reg:
                constant_init(reg.layers[-1], 0., bias=0.)
        else:
            constant_init(self.fc_reg.layers[-1], 0., bias=0.)
        if self.random_refpoints_xy:
            uniform_init(self.query_embedding)
            self.query_embedding.weight.data[:, :2] = \
                inverse_sigmoid(self.query_embedding.weight.data[:, :2])
            self.query_embedding.weight.data[:, :2].requires_grad = False

    def forward_single(self, x, img_metas):
        """Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should include background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
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
        outs_dec, reference = self.transformer(x,
                                               masks,
                                               self.query_embedding.weight,
                                               pos_embed,
                                               reg_branches=self.fc_reg)

        all_cls_scores = self.fc_cls(outs_dec)
        if not self.bbox_embed_diff_each_layer:
            # TODO: inverse_sigmoid -- to match official repo, but still slight different
            reference_before_sigmoid = inverse_sigmoid(reference,
                                                       eps=1e-3)
            tmp = self.fc_reg(outs_dec)
            tmp[..., : self.query_dim] += reference_before_sigmoid
            all_bbox_preds = tmp.sigmoid()
        else:
            reference_before_sigmoid = inverse_sigmoid(reference)
            outs_coord_lvl = []
            for lvl in range(outs_dec.size(0)):
                tmp = self.fc_reg[lvl](outs_dec[lvl])
                tmp[..., : self.query_dim] += reference_before_sigmoid[lvl]
                outs_coord_lvl.append(tmp.sigmoid())
            all_bbox_preds = torch.stack(outs_coord_lvl)
        return all_cls_scores, all_bbox_preds
