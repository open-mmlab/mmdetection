# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn.functional as F
from mmcv.cnn import Conv2d, xavier_init
from torch import nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from ..utils.positional_encoding import SinePositionalEncoding
from ..utils.transformer import DetrTransformerDecoder, DetrTransformerEncoder
from .base import BaseDetector


@DETECTORS.register_module()
class TransformerDetector(BaseDetector):

    def __init__(
            self,
            backbone,
            encoder_cfg,
            decoder_cfg,
            bbox_head,
            neck=None,
            positional_encoding_cfg=dict(num_feats=128, normalize=True),
            num_query=100,
            train_cfg=None,
            test_cfg=None,
            # pretrained=None,
            init_cfg=None):
        super(TransformerDetector, self).__init__(init_cfg)
        # if pretrained:  # TODO: Should this be deleted?
        #     warnings.warn('DeprecationWarning: pretrained is deprecated, '
        #                   'please use "init_cfg" instead')
        #     backbone.pretrained = pretrained
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.positional_encoding_cfg = positional_encoding_cfg
        self.num_query = num_query

        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.bbox_head = build_head(bbox_head)
        self._init_layers()

    def _init_layers(self):
        self._init_transformer()
        self._init_decoder_queries()
        self._init_input_proj()

    def _init_transformer(self):
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding_cfg)
        self.encoder = DetrTransformerEncoder(**self.encoder_cfg)
        self.decoder = DetrTransformerDecoder(**self.decoder_cfg)
        self.embed_dims = self.encoder.embed_dims  # TODO

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

    def _init_input_proj(self):
        in_channels = self.backbone.feat_dim  # TODO
        self.input_proj = Conv2d(in_channels, self.embed_dims, kernel_size=1)

    def _init_decoder_queries(self):
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def init_weights(self):  # TODO
        super(TransformerDetector, self).init_weights()
        self._init_transformer_weights()
        self._is_init = True  # TODO

    def _init_transformer_weights(self):  # TODO
        # follow the DetrTransformer to init parameters
        for coder in [self.encoder, self.decoder]:
            for m in coder.modules():
                if hasattr(m, 'weight') and m.weight.dim() > 1:
                    xavier_init(m, distribution='uniform')

    # def _load_from_state_dict  # TODO !

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        super(TransformerDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        assert isinstance(x, tuple) and len(x) == 1  # TODO: delete this
        x, mask, pos_embed = self.forward_pretransformer(x[0], img_metas)
        outs_dec, _ = self.forward_transformer(x, mask,
                                               self.query_embedding.weight,
                                               pos_embed)
        losses = self.bbox_head.forward_train(outs_dec, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        assert isinstance(x, tuple) and len(x) == 1  # TODO: delete this
        x, mask, pos_embed = self.forward_pretransformer(x[0], img_metas)
        outs_dec, _ = self.forward_transformer(x, mask,
                                               self.query_embedding.weight,
                                               pos_embed)
        results_list = self.bbox_head.simple_test(
            outs_dec, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        x = self.extract_feat(imgs)
        assert isinstance(x, tuple) and len(x) == 1  # TODO: delete this
        x, mask, pos_embed = self.forward_pretransformer(x[0], img_metas)
        outs_dec, _ = self.forward_transformer(x, mask,
                                               self.query_embedding.weight,
                                               pos_embed)  # TODO: may bugs
        results_list = self.bbox_head.aug_test(
            outs_dec, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def forward_pretransformer(self, x, img_metas):
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
        return x, masks, pos_embed

    def forward_transformer(self, x, mask, query_embed, pos_embed):
        bs, c, h, w = x.shape
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        x = x.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        memory = self.encoder(
            query=x, query_pos=pos_embed, query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask)
        out_dec = out_dec.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        return out_dec, memory

    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas
    def forward_dummy(self, img):
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, dummy_img_metas)
        return outs

    # over-write `onnx_export` because:
    # (1) the forward of bbox_head requires img_metas
    # (2) the different behavior (e.g. construction of `masks`) between
    # torch and ONNX model, during the forward of bbox_head
    def onnx_export(self, img, img_metas):
        """Test function for exporting to ONNX, without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        # forward of this head requires img_metas
        outs = self.bbox_head.forward_onnx(x, img_metas)
        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape

        det_bboxes, det_labels = self.bbox_head.onnx_export(*outs, img_metas)

        return det_bboxes, det_labels
