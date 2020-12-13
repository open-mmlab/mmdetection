import mmcv
import torch
import torch.nn as nn
from mmcv import tensor2imgs

from mmdet.models.builder import HEADS
from ...core import bbox_cxcywh_to_xyxy


@HEADS.register_module()
class EmbeddingRPNHead(nn.Module):

    def __init__(
        self,
        num_proposals=100,
        proposal_feature_channel=256,
        train_cfg=None,
        test_cfg=None,
    ):
        self.num_proposals = num_proposals
        self.proposal_feature_channel = proposal_feature_channel
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        super(EmbeddingRPNHead, self).__init__()
        self._init_layers()

    def _init_layers(self):
        self.init_proposal_bboxes = nn.Embedding(self.num_proposals, 4)
        self.init_proposal_features = nn.Embedding(
            self.num_proposals, self.proposal_feature_channel)

    def init_weights(self):
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)

    def _decode_init_proposals(self, imgs, img_metas):
        proposals = self.init_proposal_bboxes.weight.clone()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        num_imgs = len(imgs[0])
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(imgs[0].new_tensor([w, h, w, h])[None])
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        proposals = proposals[:, None, :] * imgs_whwh
        proposals = proposals.permute(1, 0, 2)
        init_proposal_features = self.init_proposal_features.weight.clone()

        init_proposal_features = init_proposal_features[None].expand(
            num_imgs, *init_proposal_features.size())
        return proposals, init_proposal_features, imgs_whwh

    def forward_dummy(self, img, img_metas):
        """Dummy forward function."""
        return self._decode_init_proposals(img, img_metas)

    def forward_train(self, img, img_metas):
        return self._decode_init_proposals(img, img_metas)

    def simple_test_rpn(self, img, img_metas):
        return self._decode_init_proposals(img, img_metas)

    def show_result(self, data, result, dataset=None, top_k=-1):
        assert False
        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        proposals, _ = self._decode_init_proposals(data['img'])
        assert len(imgs) == len(img_metas)
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            mmcv.imshow_bboxes(img_show, proposals, top_k=top_k)
