import torch.nn as nn

from mmdet.core import tensor2imgs, merge_aug_proposals, bbox_mapping
from .. import builder


class RPN(nn.Module):

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 rpn_train_cfg,
                 rpn_test_cfg,
                 pretrained=None):
        super(RPN, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck) if neck is not None else None
        self.rpn_head = builder.build_rpn_head(rpn_head)
        self.rpn_train_cfg = rpn_train_cfg
        self.rpn_test_cfg = rpn_test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print('load model from: {}'.format(pretrained))
        self.backbone.init_weights(pretrained=pretrained)
        if self.neck is not None:
            self.neck.init_weights()
        self.rpn_head.init_weights()

    def forward(self,
                img,
                img_meta,
                gt_bboxes=None,
                return_loss=True,
                return_bboxes=False,
                rescale=False):
        if not return_loss:
            return self.test(img, img_meta, rescale)

        img_shapes = img_meta['shape_scale']

        if self.rpn_train_cfg.get('debug', False):
            self.rpn_head.debug_imgs = tensor2imgs(img)

        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        rpn_outs = self.rpn_head(x)

        rpn_loss_inputs = rpn_outs + (gt_bboxes, img_shapes,
                                      self.rpn_train_cfg)
        losses = self.rpn_head.loss(*rpn_loss_inputs)
        return losses

    def test(self, imgs, img_metas, rescale=False):
        """Test w/ or w/o augmentations."""
        assert isinstance(imgs, list) and isinstance(img_metas, list)
        assert len(imgs) == len(img_metas)
        img_per_gpu = imgs[0].size(0)
        assert img_per_gpu == 1
        if len(imgs) == 1:
            return self.simple_test(imgs[0], img_metas[0], rescale)
        else:
            return self.aug_test(imgs, img_metas, rescale)

    def simple_test(self, img, img_meta, rescale=False):
        img_shapes = img_meta['shape_scale']
        # get feature maps
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_shapes, self.rpn_test_cfg)
        proposals = self.rpn_head.get_proposals(*proposal_inputs)[0]
        if rescale:
            proposals[:, :4] /= img_shapes[0][-1]
        return proposals.cpu().numpy()

    def aug_test(self, imgs, img_metas, rescale=False):
        aug_proposals = []
        for img, img_meta in zip(imgs, img_metas):
            x = self.backbone(img)
            if self.neck is not None:
                x = self.neck(x)
            rpn_outs = self.rpn_head(x)
            proposal_inputs = rpn_outs + (img_meta['shape_scale'],
                                          self.rpn_test_cfg)
            proposal_list = self.rpn_head.get_proposals(*proposal_inputs)
            assert len(proposal_list) == 1
            aug_proposals.append(proposal_list[0])  # len(proposal_list) = 1
        merged_proposals = merge_aug_proposals(aug_proposals, img_metas,
                                               self.rpn_test_cfg)
        if not rescale:
            img_shape = img_metas[0]['shape_scale'][0]
            flip = img_metas[0]['flip'][0]
            merged_proposals[:, :4] = bbox_mapping(merged_proposals[:, :4],
                                                   img_shape, flip)
        return merged_proposals.cpu().numpy()
