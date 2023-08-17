import copy
import warnings
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models import BaseDetector, TwoStageDetector, StandardRoIHead, SinePositionalEncoding, FCNMaskHead, \
    BaseRoIHead
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import multi_apply, unpack_gt_instances, empty_instances
from mmdet.structures import SampleList, DetDataSample
from mmdet.structures.bbox import bbox2roi
from mmdet.structures.mask import mask_target
from mmdet.utils import InstanceList, reduce_mean, OptMultiConfig
from mmdet.registry import MODELS, TASK_UTILS
from mmengine.model import BaseModel, BaseModule
from einops import rearrange, repeat
from mmdet.utils import ConfigType, OptConfigType
from mmdet.models.dense_heads import Mask2FormerHead
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead

@MODELS.register_module()
class SAMInstanceHead(Mask2FormerHead):
    def __init__(
            self,
            num_things_classes: int = 1,
            num_stuff_classes: int = 0,
            prompt_neck: ConfigType = ...,
            with_iou: bool = False,
            with_multiscale: bool = False,
            with_sincos: bool = False,
            with_res_imgfeat: bool = False,
            loss_cls: ConfigType = dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=2.0,
                reduction='mean',
                class_weight=[1.0] * 133 + [0.1]),
            loss_mask: ConfigType = dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=5.0),
            loss_dice: ConfigType = dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                reduction='mean',
                naive_dice=True,
                eps=1.0,
                loss_weight=5.0),
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            init_cfg: OptMultiConfig = None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
            **kwargs
    ):
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)

        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.with_iou = with_iou
        self.with_multiscale = with_multiscale
        self.with_sincos = with_sincos
        self.with_res_imgfeat = with_res_imgfeat

        # self.num_transformer_feat_level = num_transformer_feat_level
        # self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        # self.num_transformer_decoder_layers = transformer_decoder.num_layers
        # assert pixel_decoder.encoder.layer_cfg. \
        #            self_attn_cfg.num_levels == num_transformer_feat_level
        # pixel_decoder_ = copy.deepcopy(pixel_decoder)
        # pixel_decoder_.update(
        #     in_channels=in_channels,
        #     feat_channels=feat_channels,
        #     out_channels=out_channels)
        # self.pixel_decoder = MODELS.build(pixel_decoder_)
        # self.transformer_decoder = Mask2FormerTransformerDecoder(
        #     **transformer_decoder)
        # self.decoder_embed_dims = self.transformer_decoder.embed_dims
        #
        # self.decoder_input_projs = ModuleList()
        # # from low resolution to high resolution
        # for _ in range(num_transformer_feat_level):
        #     if (self.decoder_embed_dims != feat_channels
        #             or enforce_decoder_input_project):
        #         self.decoder_input_projs.append(
        #             Conv2d(
        #                 feat_channels, self.decoder_embed_dims, kernel_size=1))
        #     else:
        #         self.decoder_input_projs.append(nn.Identity())
        # self.decoder_positional_encoding = SinePositionalEncoding(
        #     **positional_encoding)
        # self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        # self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # # from low resolution to high resolution
        # self.level_embed = nn.Embedding(self.num_transformer_feat_level,
        #                                 feat_channels)
        #
        # self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        # self.mask_embed = nn.Sequential(
        #     nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
        #     nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
        #     nn.Linear(feat_channels, out_channels))

        self.prompt_neck = MODELS.build(prompt_neck)
        self.num_queries = self.prompt_neck.num_queries
        self.per_query_point = self.prompt_neck.per_query_point
        out_channels = self.prompt_neck.out_channels

        self.cls_embed = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 2, self.num_classes + 1)
        )

        if self.with_sincos:
            self.point_emb = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, self.per_query_point * out_channels*2)
            )
        else:
            self.point_emb = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, self.per_query_point * out_channels)
            )

        if self.with_res_imgfeat:
            self.res_imgfeat = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList,
                sam
                ) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        batch_size = len(batch_img_metas)
        decoder_out, query_feat_list, res_img_feat = self.prompt_neck(x)

        if self.with_multiscale:
            cls_pred_list = [self.cls_embed(query_feat) for query_feat in query_feat_list]
        else:
            # shape (batch_size， num_queries, c)
            cls_pred_list = [self.cls_embed(decoder_out)]
        # shape (batch_size, num_queries, c)
        point_emb = self.point_emb(decoder_out)
        # shape (batch_size, num_queries, per_query_point, c)
        point_emb = point_emb.view(batch_size, self.num_queries, self.per_query_point, -1)

        img_seg_feat = x[0]
        point_emb = rearrange(point_emb, 'b n p c -> (b n) p c')
        if self.with_sincos:
            point_emb = torch.sin(point_emb[..., ::2]) + point_emb[..., 1::2]

        nomask_dense_embeddings = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            point_emb.shape[0], -1, *img_seg_feat.shape[-2:]
        )

        img_embeddings = torch.repeat_interleave(img_seg_feat, self.num_queries, dim=0)
        img_pe = sam.prompt_encoder.get_dense_pe()
        img_pe = repeat(img_pe, 'b c h w -> (b n) c h w', n=img_embeddings.shape[0])

        if self.with_res_imgfeat:
            res_img_feat = self.res_imgfeat(res_img_feat)
            res_img_feat = torch.repeat_interleave(res_img_feat, self.num_queries, dim=0)
        else:
            res_img_feat = None

        low_res_masks, iou_predictions = sam.mask_decoder.forward_batch(
            image_embeddings=img_embeddings,
            image_pe=img_pe,
            sparse_prompt_embeddings=point_emb,
            dense_prompt_embeddings=nomask_dense_embeddings,
            multimask_output=False,
            res_img_feat=res_img_feat,
        )
        mask_pred = rearrange(low_res_masks.squeeze(1), '(b n) h w -> b n h w', b=batch_size)

        # optional
        # if self.with_iou:
        #     iou_predictions = iou_predictions.view(batch_size, self.num_queries, -1)
        #     cls_pred = cls_pred * iou_predictions

        if self.with_multiscale:
            mask_pred_list = [mask_pred] * len(cls_pred_list)
        else:
            mask_pred_list = [mask_pred]

        return cls_pred_list, mask_pred_list

    def predict(self, x: Tuple[Tensor],
                batch_data_samples: SampleList,
                sam
                ) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two tensors.

                - mask_cls_results (Tensor): Mask classification logits,\
                    shape (batch_size, num_queries, cls_out_channels).
                    Note `cls_out_channels` should includes background.
                - mask_pred_results (Tensor): Mask logits, shape \
                    (batch_size, num_queries, h, w).
        """
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        all_cls_scores, all_mask_preds = self(x, batch_data_samples, sam)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        # upsample masks
        img_shape = batch_img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        return mask_cls_results, mask_pred_results

    def loss(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        sam,
    ) -> Dict[str, Tensor]:
        """Perform forward propagation and loss calculation of the panoptic
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
            else:
                batch_gt_semantic_segs.append(None)

        # forward
        all_cls_scores, all_mask_preds = self(x, batch_data_samples, sam)

        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances,
                                                batch_gt_semantic_segs)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses


@MODELS.register_module()
class SAMAnchorInstanceHead(TwoStageDetector):
    def __init__(
            self,
            sam_head=True,
            neck: OptConfigType = None,
            rpn_head: OptConfigType = None,
            roi_head: OptConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            **kwargs
    ):
        super(TwoStageDetector, self).__init__()
        self.neck = MODELS.build(neck)
        self.sam_head = sam_head

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, x):
        x = self.neck(x)
        return x

    def loss(self,
             batch_inputs,
             batch_data_samples: SampleList,
             sam
             ) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)
        img_seg_feat = batch_inputs[0]
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        if self.sam_head:
            roi_losses = self.roi_head.loss(x, rpn_results_list,
                                            batch_data_samples,
                                            sam, img_seg_feat
                                            )
        else:
            roi_losses = self.roi_head.loss(x, rpn_results_list,
                                            batch_data_samples
                                            )
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                sam,
                rescale: bool = True
                ) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)
        img_seg_feat = batch_inputs[0]

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        if self.sam_head:
            results_list = self.roi_head.predict(
                x, rpn_results_list, batch_data_samples, sam, img_seg_feat, rescale=rescale)
        else:
            results_list = self.roi_head.predict(
                x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples


@MODELS.register_module()
class SAMAnchorPromptRoIHead(StandardRoIHead):
    def __init__(
            self,
            positional_encoding=dict(num_feats=128, normalize=True),
            *args,
            **kwargs
    ):
        super(StandardRoIHead, self).__init__(*args, **kwargs)
        self.generator_pe = SinePositionalEncoding(**positional_encoding)

    def _mask_forward(self,
                      x: Tuple[Tensor],
                      rois: Tensor = None,
                      pos_inds: Optional[Tensor] = None,
                      bbox_feats: Optional[Tensor] = None,
                      sam=None, img_seg_feat=None
                      ) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            pos_inds (Tensor, optional): Indices of positive samples.
                Defaults to None.
            bbox_feats (Tensor): Extract bbox RoI features. Defaults to None.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
        """
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_preds = self.mask_head(mask_feats, sam, img_seg_feat, img_flag_ids=rois[:, 0])
        mask_results = dict(mask_preds=mask_preds[0], mask_iou=mask_preds[1], mask_feats=mask_feats)
        return mask_results

    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], bbox_feats: Tensor,
                  batch_gt_instances: InstanceList,
                  sam, img_seg_feat
                  ) -> dict:
        """Perform forward propagation and loss calculation of the mask head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            bbox_feats (Tensor): Extract bbox RoI features.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
                - `mask_targets` (Tensor): Mask target of each positive\
                    proposals in the image.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            mask_results = self._mask_forward(
                x, pos_rois, sam=sam, img_seg_feat=img_seg_feat)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)

        mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])
        return mask_results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample],
             sam, img_seg_feat
             ) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        x = list(x)
        bs, _, h, w = x[-1].shape
        mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
        img_feats_pe = self.generator_pe(mask_pe)
        for i in range(len(x)):
            x[i] = x[i] + torch.nn.functional.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear')

        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances,
                                          sam, img_seg_feat
                                          )
            losses.update(mask_results['loss_mask'])

        return losses


    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False,
                     sam=None, img_seg_feat=None
                     ) -> InstanceList:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        mask_results = self._mask_forward(x, mask_rois, sam=sam, img_seg_feat=img_seg_feat)
        mask_preds = mask_results['mask_preds']
        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

        # TODO: Handle the case where rescale is false
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        return results_list

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                sam, img_seg_feat,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (N, C, H, W).
            rpn_results_list (list[:obj:`InstanceData`]): list of region
                proposals.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results to
                the original image. Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        x = list(x)
        bs, _, h, w = x[-1].shape
        mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
        img_feats_pe = self.generator_pe(mask_pe)
        for i in range(len(x)):
            x[i] = x[i] + torch.nn.functional.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear')

        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        # TODO: nms_op in mmcv need be enhanced, the bbox result may get
        #  difference when not rescale in bbox_head

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale)

        if self.with_mask:
            results_list = self.predict_mask(
                x, batch_img_metas, results_list, rescale=rescale, sam=sam, img_seg_feat=img_seg_feat)

        return results_list


@MODELS.register_module()
class SAMPromptMaskHead(FCNMaskHead):

    def __init__(self,
                 per_query_point: int = 5,
                 with_sincos: bool = True,
                 class_agnostic: bool = False,
                 loss_mask: ConfigType = dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 *args,
                 **kwargs
                 ) -> None:
        super(BaseModule, self).__init__()

        self.per_query_point = per_query_point
        self.with_sincos = with_sincos
        self.class_agnostic = class_agnostic

        self.loss_mask = MODELS.build(loss_mask)

        if with_sincos:
            sincos = 2
        else:
            sincos = 1
        self.point_emb = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(7*7*256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256*sincos*per_query_point)
        )

    def forward(self, x, sam, img_seg_feat, img_flag_ids) -> Tensor:
        batch_size = x.shape[0]
        point_emb = self.point_emb(x)
        point_emb = point_emb.view(batch_size, self.per_query_point, -1)
        if self.with_sincos:
            point_emb = torch.sin(point_emb[..., ::2]) + point_emb[..., 1::2]

        nomask_dense_embeddings = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            point_emb.shape[0], -1, *img_seg_feat.shape[-2:]
        )
        img_flag_ids = torch.bincount(img_flag_ids.long())
        padding = torch.zeros((len(img_seg_feat)-len(img_flag_ids),), device=img_flag_ids.device, dtype=img_flag_ids.dtype)
        img_flag_ids = torch.cat([img_flag_ids, padding])
        img_embeddings = torch.repeat_interleave(img_seg_feat, img_flag_ids, dim=0)
        img_pe = sam.prompt_encoder.get_dense_pe()
        img_pe = repeat(img_pe, 'b c h w -> (b n) c h w', n=img_embeddings.shape[0])

        res_img_feat = None
        low_res_masks, iou_predictions = sam.mask_decoder.forward_batch(
            image_embeddings=img_embeddings,
            image_pe=img_pe,
            sparse_prompt_embeddings=point_emb,
            dense_prompt_embeddings=nomask_dense_embeddings,
            multimask_output=False,
            res_img_feat=res_img_feat,
        )
        mask_pred = low_res_masks.squeeze(1)
        iou_predictions = iou_predictions.squeeze(1)
        return mask_pred, iou_predictions

    def get_targets(self, sampling_results: List[SamplingResult],
                    batch_gt_instances: InstanceList,
                    rcnn_train_cfg: ConfigDict) -> Tensor:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.

        Returns:
            Tensor: Mask target of each positive proposals in the image.
        """
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        gt_masks = [res.masks for res in batch_gt_instances]

        mask_targets_list = []
        mask_size = (rcnn_train_cfg.mask_size,) * 2
        device = pos_proposals[0].device
        for pos_gt_inds, gt_mask in zip(pos_assigned_gt_inds, gt_masks):
            if len(pos_gt_inds) == 0:
                mask_targets = torch.zeros((0,) + mask_size, device=device, dytpe=torch.float32)
            else:
                mask_targets = gt_mask[pos_gt_inds.cpu()].to_tensor(dtype=torch.float32, device=device)
            mask_targets_list.append(mask_targets)
        mask_targets = torch.cat(mask_targets_list)
        return mask_targets

    def loss_and_target(self, mask_preds: Tensor,
                        sampling_results: List[SamplingResult],
                        batch_gt_instances: InstanceList,
                        rcnn_train_cfg: ConfigDict) -> dict:
        """Calculate the loss based on the features extracted by the mask head.

        Args:
            mask_preds (Tensor): Predicted foreground masks, has shape
                (num_pos, num_classes, h, w).
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.

        Returns:
            dict: A dictionary of loss and targets components.
        """
        mask_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        mask_preds = torch.nn.functional.interpolate(
            mask_preds.unsqueeze(1), size=mask_targets.shape[-2:], mode='bilinear', align_corners=False)
        loss = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_preds, mask_targets,
                                           torch.zeros_like(pos_labels))
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets,
                                           pos_labels)
        loss['loss_mask'] = loss_mask
        # TODO: which algorithm requires mask_targets?
        return dict(loss_mask=loss, mask_targets=mask_targets)

    def _predict_by_feat_single(self,
                                mask_preds: Tensor,
                                bboxes: Tensor,
                                labels: Tensor,
                                img_meta: dict,
                                rcnn_test_cfg: ConfigDict,
                                rescale: bool = False,
                                activate_map: bool = False) -> Tensor:
        """Get segmentation masks from mask_preds and bboxes.

        Args:
            mask_preds (Tensor): Predicted foreground masks, has shape
                (n, num_classes, h, w).
            bboxes (Tensor): Predicted bboxes, has shape (n, 4)
            labels (Tensor): Labels of bboxes, has shape (n, )
            img_meta (dict): image information.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            activate_map (book): Whether get results with augmentations test.
                If True, the `mask_preds` will not process with sigmoid.
                Defaults to False.

        Returns:
            Tensor: Encoded masks, has shape (n, img_w, img_h)

        Example:
            >>> from mmengine.config import Config
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> self = FCNMaskHead(num_classes=C, num_convs=0)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_preds = self.forward(inputs)
            >>> # Each input is associated with some bounding box
            >>> bboxes = torch.Tensor([[1, 1, 42, 42 ]] * N)
            >>> labels = torch.randint(0, C, size=(N,))
            >>> rcnn_test_cfg = Config({'mask_thr_binary': 0, })
            >>> ori_shape = (H * 4, W * 4)
            >>> scale_factor = (1, 1)
            >>> rescale = False
            >>> img_meta = {'scale_factor': scale_factor,
            ...             'ori_shape': ori_shape}
            >>> # Encoded masks are a list for each category.
            >>> encoded_masks = self._get_seg_masks_single(
            ...     mask_preds, bboxes, labels,
            ...     img_meta, rcnn_test_cfg, rescale)
            >>> assert encoded_masks.size()[0] == N
            >>> assert encoded_masks.size()[1:] == ori_shape
        """
        scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            (1, 2))
        img_h, img_w = img_meta['ori_shape'][:2]
        device = bboxes.device

        if not activate_map:
            mask_preds = mask_preds.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_preds = bboxes.new_tensor(mask_preds)

        if rescale:  # in-placed rescale the bboxes
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)

        threshold = rcnn_test_cfg.mask_thr_binary

        im_mask = torch.nn.functional.interpolate(
            mask_preds.unsqueeze(1), size=(img_h, img_w), mode='bilinear', align_corners=False).squeeze(1)

        if threshold >= 0:
            im_mask = im_mask >= threshold
        else:
            # for visualization and debugging
            im_mask = (im_mask * 255).to(dtype=torch.uint8)
        return im_mask