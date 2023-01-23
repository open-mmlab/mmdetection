import warnings
import torch
import torch.nn.functional as F
from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS, build_detector
from mmdet.models.detectors.fcos import FCOS


@DETECTORS.register_module()
class FCOS_TS(FCOS):
    def __init__(self,
                 distill_type,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_cfg,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

        # Teacher Network
        teacher_cfg.model.type = 'FCOS_Cont'
        teacher_cfg.model.bbox_head.type = 'FCOSHead_Cont'
        teacher_cfg.model.bbox_head.output_roi_size = bbox_head.output_roi_size
        self.teacher_cfg = teacher_cfg
        
        self.distill_type = distill_type
        
    def update_teacher(self, state_dict): 
        # Load Teacher Model
        self.teacher = build_detector(self.teacher_cfg.model,
                                      train_cfg=None,
                                      test_cfg=None)
        
        # Load Pretrained Teacher Weights
        self.teacher.load_state_dict(state_dict, strict=True)
        
        # Freeze Param
        for param in self.teacher.parameters():
            param.requires_grad = False
        
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        
        x = self.extract_feat(img)
        losses, gt_reg_feat, gt_cls_feat = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        return losses, gt_reg_feat, gt_cls_feat


    def train_step(self, data, optimizer):
        # Teacher Feat
        self.teacher.eval()
        with torch.no_grad():
            _, gt_reg_feat_teacher, gt_cls_feat_teacher = self.teacher(**data[0])
            
        # Student Feat
        losses, gt_reg_feat_student, gt_cls_feat_student = self(**data[1])
        
        # Calc Consistency Loss
        B = gt_reg_feat_teacher.size(0)
        if self.distill_type == 'reg':
            gt_reg_feat_teacher = gt_reg_feat_teacher.view(B, -1)
            gt_reg_feat_student = gt_reg_feat_student.view(B, -1)            
            consistency_loss = self.calc_consistency_loss(gt_reg_feat_teacher, gt_reg_feat_student)
            
        elif self.distill_type == 'cls':
            gt_cls_feat_teacher = gt_cls_feat_teacher.view(B, -1)
            gt_cls_feat_student = gt_cls_feat_student.view(B, -1)            
            consistency_loss = self.calc_consistency_loss(gt_cls_feat_teacher, gt_cls_feat_student)
            
        elif self.distill_type == 'both':
            gt_reg_feat_teacher = gt_reg_feat_teacher.view(B, -1)
            gt_reg_feat_student = gt_reg_feat_student.view(B, -1)            
            consistency_loss_reg = self.calc_consistency_loss(gt_reg_feat_teacher, gt_reg_feat_student)

            gt_cls_feat_teacher = gt_cls_feat_teacher.view(B, -1)
            gt_cls_feat_student = gt_cls_feat_student.view(B, -1)            
            consistency_loss_cls = self.calc_consistency_loss(gt_cls_feat_teacher, gt_cls_feat_student)
            
            consistency_loss = (consistency_loss_reg + consistency_loss_cls) / 2
            
        else:
            raise('Select Proper Distill Type')
        
        losses.update({'consistency_loss': consistency_loss * 1.0})
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data[0]['img_metas']))

        return outputs
    
    def calc_consistency_loss(self, feat_ori, feat_aug):
        return torch.mean(1.0 - F.cosine_similarity(feat_ori, feat_aug))
    
    
    
@DETECTORS.register_module()
class FCOS_Cont(FCOS):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        
        x = self.extract_feat(img)
        losses, reg_feat, cls_feat = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        return losses, reg_feat, cls_feat