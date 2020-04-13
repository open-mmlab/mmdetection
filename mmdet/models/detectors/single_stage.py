import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):#继承BASEDETECTOR，单阶段模型
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)#主干，所有基于builder的函数都是直接使用build_from_config这个函数进行建立，
        if neck is not None:                             #返回的就是obj_cls(**args)
            self.neck = builder.build_neck(neck)#neck部分
        self.bbox_head = builder.build_head(bbox_head)#head部分
        #分离出train,test两个部分的参数字典
        self.train_cfg = train_cfg                               
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)#主干进行预训练初始化
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):#对neck的部分使用其他的初始化方式
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):#抽取图片特征，就是网络的执行
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)#这个是比抽取特征加上head，就是backbone+neck+bbox_head
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):#这个是用于训练的前向，是对基类的重写，不同于二阶段分类器，它更加简单
        x = self.extract_feat(img)#直接抽取特征
        outs = self.bbox_head(x)#这里其实相当于forward_dummy所做的事情
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)#将结果和GT放在一起共同进入loss
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):#重写基类，是对于一张图的测试
        x = self.extract_feat(img)#抽取特征
        outs = self.bbox_head(x)#加上bbox_head，就是输出的结果
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]#生成的结果
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
