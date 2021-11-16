#TODO maskformer head
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.runner import ModuleList
# from mmdet.models import losses

from ...builder import HEADS
# from ..base_semantic_head import BaseSemanticHead
from ...utils import HungarianMatcher

# det2
from .pixel_decoder import TransformerEncoderPixelDecoder
from .transformer_predictor import TransformerPredictor
from .criterion import SetCriterion 
from .utils import sem_seg_postprocess
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
import os 
import os.path as osp 
import pickle as pkl 


@HEADS.register_module()
class MaskFormerHead(nn.Module):
    """ 
    
    transformer decoder * 6
    pixel_decoder: "fpn" + transformer encoder * 6

    """

    def __init__(self,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 in_channels=[256, 512, 1024, 2048],
                 conv_dim=256,
                 mask_dim=256,
                 num_queries=100,
                 num_head=8,
                 sem_seg_postprocess_before_inference=True,
                 init_cfg=None):
        # super(MaskFormerHead, self).__init__(num_stuff_classes + 1, init_cfg,
                                            #   loss_seg)
        super(MaskFormerHead, self).__init__()
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.object_mask_threshold = 0.8
        self.overlap_threshold = 0.8
        self.panoptic_on = True
        #! thing 在前 0-79， stuff在后 80-132,  # 255 as ignore
        self.mask_scale_factor = 4
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        ignore_value = -1
        loss_weight = 1.0
        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight
        self.pixel_decoder = TransformerEncoderPixelDecoder(in_channels, 
                                                            transformer_dropout=0.1,
                                                            transformer_nheads=num_head,
                                                            transformer_dim_feedforward=2048,
                                                            transformer_enc_layers=6,
                                                            transformer_pre_norm=False,
                                                            conv_dim=conv_dim,
                                                            mask_dim=mask_dim,
                                                            norm="GN")
        self.predictor = TransformerPredictor(in_channels[0],
                                              mask_classification=True,
                                              num_classes=self.num_classes,
                                              hidden_dim=conv_dim,
                                              num_queries=num_queries,
                                              nheads=num_head,
                                              dropout=0.1,
                                              dim_feedforward=2048,
                                              enc_layers=0,
                                              dec_layers=6,
                                              pre_norm=False,
                                              deep_supervision=True,
                                              mask_dim=mask_dim,
                                              enforce_input_project=False)

        self.transformer_in_feature = "transformer_encoder"

        matcher = HungarianMatcher(cost_class=1, cost_mask=20.0, cost_dice=1.0)
        weight_dict = {'loss_ce': 1, 
                       'loss_mask': 20.0, 
                       'loss_dice': 1.0, 
                       'loss_ce_0': 1, 
                       'loss_mask_0': 20.0, 
                       'loss_dice_0': 1.0, 
                       'loss_ce_1': 1,
                       'loss_mask_1': 20.0, 
                       'loss_dice_1': 1.0, 
                       'loss_ce_2': 1, 
                       'loss_mask_2': 20.0, 
                       'loss_dice_2': 1.0, 
                       'loss_ce_3': 1, 
                       'loss_mask_3': 20.0, 
                       'loss_dice_3': 1.0, 
                       'loss_ce_4': 1, 
                       'loss_mask_4': 20.0, 
                       'loss_dice_4': 1.0}
        self.criterion = SetCriterion(num_classes=self.num_classes, 
                                      matcher=matcher, 
                                      weight_dict=weight_dict, 
                                      eos_coef=0.1, 
                                      losses=['labels', 'masks'])


    def prepare_target(self, gt_labels, gt_masks, gt_semantic_seg):
        """
        Args:
            gt_bboxes (list[Tensor]): element's shape = (num_objects, 4).
            gt_labels (list[Tensor]): element's shape = (num_objects, ).
            gt_masks (list[BitmapMasks]): element's shape = (num_objects, img_h, img_w).
            gt_semantic_seg (Tensor): shape = (B, img_h / 4, img_w / 4).

        Returns:
            list[Dict[str, Tensor]]: labels, masks
        """
        targets = []
        batch_size = len(gt_labels)
        for i in range(batch_size):
            target = self._prepare_single_target(gt_labels=gt_labels[i], 
                gt_masks=gt_masks[i], gt_semantic_seg=gt_semantic_seg[i])
            targets.append(target)
        
        return targets


    def _prepare_single_target(self, gt_labels, gt_masks, gt_semantic_seg):
        """
        Args:
            gt_labels (Tensor): shape = (num_objects, ), dtype=int64
            gt_masks (Tensor): shape = (num_objects, img_h, img_w)
            gt_semantic_seg (Tensor): shape = (img_h / 4, img_w / 4), 
                thing 在前 0-79， stuff在后 80-132, 255为ignore. 

        Returns:
            dict[str, Tensor]: labels, masks
        """
        things_labels = gt_labels
        things_masks = gt_masks.rescale(self.mask_scale_factor).to_tensor(dtype=torch.bool, device=gt_labels.device)
        assert things_masks.shape[-2:] == gt_semantic_seg.shape[-2:]
        stuff_semantic_seg = torch.masked_select(gt_semantic_seg, 
                                self.num_things_classes <= gt_semantic_seg and 
                                gt_semantic_seg < (self.num_stuff_classes + self.num_things_classes))
        stuff_labels = torch.unique(stuff_semantic_seg, sorted=False, return_inverse=False, return_counts=False)
        stuff_masks = []
        for label in stuff_labels:
            stuff_mask = gt_semantic_seg == label 
            stuff_masks.append(stuff_mask)
        stuff_masks = torch.stack(stuff_masks, dim=0)
        labels = torch.cat([things_labels, stuff_labels])
        masks = torch.cat([things_masks, stuff_masks])

        target = {
            "labels": labels,
            "masks": masks
        }        

        return target


    def loss(self, label_preds, seg_preds, gt_labels, gt_masks):

        loss = dict()
        
        return loss

    
    def forward(self, feats):
        mask_features, transformer_encoder_features = self.pixel_decoder.forward_features(feats)
        # mask_features torch.Size([1, 256, 200, 302])
        # transformer_encoder_features torch.Size([1, 256, 25, 38])
        # print("mask_features", mask_features.shape)
        # print("transformer_encoder_features", transformer_encoder_features.shape)
        if self.transformer_in_feature == "transformer_encoder":
            assert (
                transformer_encoder_features is not None
            ), "Please use the TransformerEncoderPixelDecoder."
            predictions = self.predictor(transformer_encoder_features, mask_features)
        else:
            #TODO transformer_in_feature = res5
            predictions = self.predictor(feats[-1], mask_features)
        return predictions


    def forward_train(self, 
                      feats, 
                      img_metas, 
                      gt_bboxes, 
                      gt_labels, 
                      gt_masks, 
                      gt_semantic_seg,
                      gt_bboxes_ignore=None):
        """
        Args:
            x (Tensor): feature map of shape = (B, C, H, W).
            img_metas (list[Dict]): image metas.
            gt_bboxes (list[Tensor]): shape of each element = (num_objects, 4).
            gt_labels (list[Tensor]): shape of each element = (num_objects, ).
            gt_masks (list[BitmapMasks]): 表示不同的实例的mask.
            gt_semantic_seg (list[tensor]): thing 在前 0-79， stuff在后 80-132, 255为背景. 
            gt_bboxes_ignore (list[Tensor]): # TODO 不知道是否需要考虑掉这些实例. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # prepare target
        targets = self.prepare_target(gt_labels, gt_masks, gt_semantic_seg)

        # forward
        outs = self(feats)     

        # loss
        losses = self.criterion(outs, targets)
        
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses


    def simple_test(self, feats, img_metas, rescale=False):
        outputs = self(feats)
        # ['pred_logits', 'pred_masks', 'aux_outputs']
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        # 000000000139
        # if (img_metas[0]["ori_filename"].replace(".jpg","") == "000000000139"):
        #     pkl_path = "./000000000139.pkl"
        #     with open(pkl_path, "wb") as f:
        #         pkl.dump({"pred_logits": mask_cls_results[0].detach().cpu().numpy(),
        #             "pred_masks": mask_pred_results[0].detach().cpu().numpy()}, f)
        #     exit(-1)
        # upsample masks
        img_shape = img_metas[0]["pad_shape"][:2]
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode="bilinear",
            align_corners=False,
        )

        processed_results = []
        for mask_cls_result, mask_pred_result, meta in zip(
            mask_cls_results, mask_pred_results, img_metas):
            # padding 之前的输入图像大小
            height, width = meta["ori_shape"][:2]
            # height, width = meta["img_shape"][:2]
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = sem_seg_postprocess(
                    mask_pred_result, img_shape, height, width
                )

            # semantic segmentation inference
            r = self.semantic_inference(mask_cls_result, mask_pred_result)
            if not self.sem_seg_postprocess_before_inference:
                r = sem_seg_postprocess(r, img_shape, height, width)
            # processed_results.append({"sem_seg": r.detach().cpu().numpy()})

            # panoptic segmentation inference
            if self.panoptic_on:
                # print(mask_pred_result.shape, "xxxxxxxxxxxxxxxxx")
                panoptic_r = self.panoptic_inference(mask_cls_result, mask_pred_result)
                processed_results.append({"pan_results": panoptic_r.detach().cpu().numpy()})

            # print(r.shape, r.dtype, panoptic_r.shape, panoptic_r.dtype)
        # print(processed_results[-1]["pan_results"].shape)
        return processed_results

    
    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    
    def panoptic_inference(self, mask_cls, mask_pred):
        """ 
        Args:
            mask_cls (Tensor): shape = [N, ], N: num_queries
            mask_pred (Tensor): shape = [N, H, W], N: num_queries

        Returns:
            dict[str, Tensor]: {"pan_results": tensor of shape = (H, W) and dtype=int32},
            each element in Tensor means: segment_id = _cls + instance_id * INSTANCE_OFFSET.
        """
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device).fill_(self.num_classes)

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            # take argmax
            # cur_prob_masks.shape = (N, H, W)
            cur_mask_ids = cur_prob_masks.argmax(0)
            instance_id = 1
            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                isthing = pred_class < self.num_things_classes
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    if not isthing:
                        # different stuff regions of same class will be merged here,
                        # and stuff share the instance_id 0.
                        panoptic_seg[mask] = pred_class
                    else:
                        panoptic_seg[mask] = pred_class + instance_id * INSTANCE_OFFSET
                        instance_id += 1
        return panoptic_seg
