# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmdet.models.builder import HEADS
from ...core import bbox_cxcywh_to_xyxy


@HEADS.register_module()
class EmbeddingRPNHead(BaseModule):
    """RPNHead in the `Sparse R-CNN <https://arxiv.org/abs/2011.12450>`_ .

    Unlike traditional RPNHead, this module does not need FPN input, but just
    decode `init_proposal_bboxes` and expand the first dimension of
    `init_proposal_bboxes` and `init_proposal_features` to the batch_size.

    Args:
        num_proposals (int): Number of init_proposals. Default 100.
        proposal_feature_channel (int): Channel number of
            init_proposal_feature. Defaults to 256.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_proposals=100,
                 proposal_feature_channel=256,
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(EmbeddingRPNHead, self).__init__(init_cfg)
        self.num_proposals = num_proposals
        self.proposal_feature_channel = proposal_feature_channel
        self._init_layers()

    def _init_layers(self):
        """Initialize a sparse set of proposal boxes and proposal features."""
        self.init_proposal_bboxes = nn.Embedding(self.num_proposals, 4)
        self.init_proposal_features = nn.Embedding(
            self.num_proposals, self.proposal_feature_channel)

    def init_weights(self):
        """Initialize the init_proposal_bboxes as normalized.

        [c_x, c_y, w, h], and we initialize it to the size of  the entire
        image.
        """
        super(EmbeddingRPNHead, self).init_weights()
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)

    def _decode_init_proposals(self, imgs, img_metas):
        """Decode init_proposal_bboxes according to the size of images and
        expand dimension of init_proposal_features to batch_size.

        Args:
            imgs (list[Tensor]): List of FPN features.
            img_metas (list[dict]): List of meta-information of
                images. Need the img_shape to decode the init_proposals.

        Returns:
            Tuple(Tensor):

                - proposals (Tensor): Decoded proposal bboxes,
                  has shape (batch_size, num_proposals, 4).
                - init_proposal_features (Tensor): Expanded proposal
                  features, has shape
                  (batch_size, num_proposals, proposal_feature_channel).
                - imgs_whwh (Tensor): Tensor with shape
                  (batch_size, 4), the dimension means
                  [img_width, img_height, img_width, img_height].
        """
        proposals = self.init_proposal_bboxes.weight.clone()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        num_imgs = len(imgs[0])
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(imgs[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        # imgs_whwh has shape (batch_size, 1, 4)
        # The shape of proposals change from (num_proposals, 4)
        # to (batch_size ,num_proposals, 4)
        proposals = proposals * imgs_whwh

        init_proposal_features = self.init_proposal_features.weight.clone()
        init_proposal_features = init_proposal_features[None].expand(
            num_imgs, *init_proposal_features.size())
        return proposals, init_proposal_features, imgs_whwh

    def forward_dummy(self, img, img_metas):
        """Dummy forward function.

        Used in flops calculation.
        """
        return self._decode_init_proposals(img, img_metas)

    def forward_train(self, img, img_metas):
        """Forward function in training stage."""
        return self._decode_init_proposals(img, img_metas)

    def simple_test_rpn(self, img, img_metas):
        """Forward function in testing stage."""
        return self._decode_init_proposals(img, img_metas)

    def simple_test(self, img, img_metas):
        """Forward function in testing stage."""
        raise NotImplementedError

    def aug_test_rpn(self, feats, img_metas):
        raise NotImplementedError(
            'EmbeddingRPNHead does not support test-time augmentation')
