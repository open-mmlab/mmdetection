import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_plugin_layer, kaiming_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import force_fp32

from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from ..builder import HEADS, build_loss
from ..dense_heads.anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class MaskFormerHead(AnchorFreeHead):
    """Implements the MaskFormer head.

    See `paper: Per-Pixel Classification is Not All You Need
    for Semantic Segmentation<https://arxiv.org/pdf/2107.06278>`
    for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number channels for feature.
        out_channels (int): Number channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer.
        pixel_decoder (obj:`mmcv.ConfigDict`|dict): Config for pixel decoder.
            Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add a layer
            to change the embed_dim of tranformer encoder in pixel decoder to
            the embed_dim of transformer decoder. Defaults to False.
        transformer_decoder (obj:`mmcv.ConfigDict`|dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (obj:`mmcv.ConfigDict`|dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the classification
            loss. Defaults to `CrossEntropyLoss`.
        loss_mask (obj:`mmcv.ConfigDict`|dict): Config of the mask loss.
            Defaults to `FocalLoss`.
        loss_dice (obj:`mmcv.ConfigDict`|dict): Config of the dice loss.
            Defaults to `DiceLoss`.
        train_cfg (obj:`mmcv.ConfigDict`|dict): Training config of Maskformer
            head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of Maskformer
            head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_queries=100,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_mask=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=20.0),
                 loss_dice=dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     naive_dice=True,
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries

        pixel_decoder.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        pixel_decoder_name, layer = build_plugin_layer(pixel_decoder)
        self.pixel_decoder_name = pixel_decoder_name
        self.add_module(pixel_decoder_name, layer)
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        pixel_decoder_type = pixel_decoder.get('type', None)
        if pixel_decoder_type == 'PixelDecoder' and (
                self.decoder_embed_dims != in_channels[-1]
                or enforce_decoder_input_project):
            self.decoder_input_proj = Conv2d(
                in_channels[-1], self.decoder_embed_dims, kernel_size=1)
        else:
            self.decoder_input_proj = nn.Identity()
        self.decoder_pe = build_positional_encoding(positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, out_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = build_assigner(assigner)
            sampler_cfg = dict(type='MaskPseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.bg_cls_weight = 0
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is MaskFormerHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official MaskFormerHead repo, bg_cls_weight
            # means relative classification weight of the VOID class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(self.num_classes + 1) * class_weight
            # set VOID class as the last indice
            class_weight[self.num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

    def init_weights(self):
        kaiming_init(self.decoder_input_proj, a=1)

    @property
    def pixel_decoder(self):
        return getattr(self, self.pixel_decoder_name)

    def preprocess_gt(self, gt_labels_list, gt_masks_list, gt_semantic_segs):
        """Preprocess the ground truth for all images:

            - labels should contain labels for each type of stuff and
                labels for each instance.
            - masks sholud contain masks for each type of stuff and
                masks for each instance.

        Args:
            gt_labels_list (list[Tensor]): Each is ground truth
                labels of each bbox, with shape [num_gts, ].
            gt_masks_list (list[Tensor]): Each is ground truth
                masks of each instances of a image, shape
                (num_gts, h, w).
            gt_semantic_seg (Tensor): Ground truth of semantic
                segmentation with the shape (bs, n, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID.
            target_shape (tuple[int]): Shape of output mask_preds.
                Resize the masks to shape of mask_preds.

        Returns:
            tuple: a tuple containing the following targets.

                - labels (list[Tensor]): Ground truth class indices for all
                    images. Each with shape (n, ), n is the sum of number
                    of stuff type and number of instance in a image.
                - masks (list[Tensor]): Ground truth mask for each image, each
                    with shape (n, h, w).
        """
        targets = multi_apply(self._preprocess_gt_single, gt_labels_list,
                              gt_masks_list, gt_semantic_segs)
        labels, masks = targets
        return labels, masks

    def _preprocess_gt_single(self, gt_labels, gt_masks, gt_semantic_seg):
        """Preprocess the ground truth for a image:

            - labels should contain labels for each type of stuff and
                labels for each instance.
            - masks sholud contain masks for each type of stuff and
                masks for each instance.

        Args:
            gt_labels (Tensor): Ground truth labels of each bbox,
                with shape [num_gts, ].
            gt_masks (Tensor): Ground truth masks of each instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (Tensor): Ground truth of semantic
                segmentation with the shape (1, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID.
            target_shape (tuple[int]): Shape of output mask_preds.
                Resize the masks to shape of mask_preds.

        Returns:
            tuple: a tuple containing the following targets.

                - labels (Tensor): Ground truth class indices for a
                    image, with shape (n, ), n is the sum of number
                    of stuff type and number of instance in a image.
                - masks (Tensor): Ground truth mask for a image, with
                    shape (n, h, w).
        """
        things_labels = gt_labels
        gt_semantic_seg = gt_semantic_seg.squeeze(0)

        things_masks = gt_masks.pad(gt_semantic_seg.shape[-2:], pad_val=0)\
            .to_tensor(dtype=torch.bool, device=gt_labels.device)

        semantic_labels = torch.unique(
            gt_semantic_seg,
            sorted=False,
            return_inverse=False,
            return_counts=False)
        stuff_masks_list = []
        stuff_labels_list = []
        for label in semantic_labels:
            if label < self.num_things_classes or label >= self.num_classes:
                continue
            stuff_mask = gt_semantic_seg == label
            stuff_masks_list.append(stuff_mask)
            stuff_labels_list.append(label)

        if len(stuff_masks_list) > 0:
            stuff_masks = torch.stack(stuff_masks_list, dim=0)
            stuff_labels = torch.stack(stuff_labels_list, dim=0)
            labels = torch.cat([things_labels, stuff_labels], dim=0)
            masks = torch.cat([things_masks, stuff_masks], dim=0)
        else:
            labels = things_labels
            masks = things_masks

        masks = masks.long()
        return labels, masks

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape [num_query,
                cls_out_channels].
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape [num_query, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels of all images.
                    Each with shape [num_query, ].
                - label_weights_list (list[Tensor]): Label weights of all
                    images.Each with shape [num_query, ].
                - mask_targets_list (list[Tensor]): Mask targets of all images.
                    Each with shape [num_query, h, w].
                - mask_weights_list (list[Tensor]): Mask weights of all images.
                    Each with shape [num_query, ].
                - num_total_pos (int): Number of positive samples in all
                    images.
                - num_total_neg (int): Number of negative samples in all
                    images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks,
                           img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape [num_query, h, w].
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (n, ). n is the sum of number of stuff type and number
                of instance in a image.
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (n, h, w).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                    shape [num_query, ].
                - label_weights (Tensor): Label weights of each image.
                    shape [num_query, ].
                - mask_targets (Tensor): Mask targets of each image.
                    shape [num_query, h, w].
                - mask_weights (Tensor): Mask weights of each image.
                    shape [num_query, ].
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        target_shape = mask_pred.shape[-2:]
        gt_masks_downsampled = F.interpolate(
            gt_masks.unsqueeze(1).float(), target_shape,
            mode='nearest').squeeze(1).long()
        # assign and sample
        assign_result = self.assigner.assign(cls_score, mask_pred, gt_labels,
                                             gt_masks_downsampled, img_metas)
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(self.num_queries)

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = torch.zeros((self.num_queries, ),
                                   dtype=torch.float32,
                                   device=mask_pred.device)
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
             gt_masks_list, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape [nb_dec, bs, num_query, cls_out_channels].
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape [n_dec, bs, num_query, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape [bs, num_query, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (n, ). n is the sum of number of stuff
                types and number of instances in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]:Loss components for outputs from a single decoder
                layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas)

        labels = torch.stack(labels_list, dim=0)  # shape [bs, bq]
        label_weights = torch.stack(
            label_weights_list, dim=0)  # shape [bs, bq]
        # ! shape [n_gts, h, w]
        mask_targets = torch.cat(mask_targets_list, dim=0)
        mask_weights = torch.stack(
            mask_weights_list, dim=0)  # ! shape [bs, nq]

        # classfication loss
        cls_scores = cls_scores.flatten(0, 1)  # shape [bs * nq, ]
        labels = labels.flatten(0, 1)  # shape [bs * nq, ]
        label_weights = label_weights.flatten(0, 1)  # shape [bs* nq, ]

        class_weight = torch.ones(self.num_classes + 1, device=labels.device)
        class_weight[-1] = self.bg_cls_weight
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = torch.tensor(
                0.0, dtype=torch.float32, device=mask_preds.device)
            loss_mask = torch.tensor(
                0.0, dtype=torch.float32, device=mask_preds.device)
            return loss_cls, loss_mask, loss_dice

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones and upsample
        mask_preds = mask_preds[mask_weights > 0]
        target_shape = mask_targets.shape[-2:]
        mask_preds = F.interpolate(
            mask_preds.unsqueeze(1),
            target_shape,
            mode='bilinear',
            align_corners=False).squeeze(1)  # [n_gts, h, w]

        # dice loss
        loss_dice = self.loss_dice(
            mask_preds, mask_targets, avg_factor=num_total_masks)

        # mask loss
        # ! FocalLoss support input of shape [n, num_class]
        h, w = mask_preds.shape[-2:]
        # [n_gts, h, w] -> [n_gts * h * w, 1]
        mask_preds = mask_preds.reshape(-1, 1)
        # [n_gts, h, w] -> [n_gts * h * w]
        mask_targets = mask_targets.reshape(-1)
        # ! 1 - mask_targets
        loss_mask = self.loss_mask(
            mask_preds, 1 - mask_targets, avg_factor=num_total_masks * h * w)

        return loss_cls, loss_mask, loss_dice

    def forward(self, feats):
        """Forward function.

        Args:
            feats (list[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            all_cls_scores (Tensor): Classification scores for each
                scale level. Each is a 4D-tensor with shape
                    [nb_dec, bs, num_query, cls_out_channels].
                 Note `cls_out_channels` should includes background.
            all_mask_preds (Tensor): Mask scores for each decoder
                layer. Each with shape [n_dec, bs, num_query, h, w].
        """
        bs, c, h, w = feats[-1].shape
        # when backbone is swin, memory is output of last stage of swin.
        # when backbone is r50, memory is output of tranformer encoder.
        mask_features, memory = self.pixel_decoder(feats)
        padding_mask = feats[-1].new_zeros((bs, h, w), dtype=torch.bool)
        pos_embed = self.decoder_pe(padding_mask)
        memory = self.decoder_input_proj(memory)
        # [bs, c, h, w] -> [h*w, bs, c]
        memory = memory.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        padding_mask = padding_mask.flatten(1)  # shape [bs, h * w]
        query_embed = self.query_embed.weight  # shape = [nq, em]
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # shape = [nq, bs, em]
        target = torch.zeros_like(query_embed)
        # [n_dec, nq, bs, em]
        out_dec = self.transformer_decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=padding_mask)
        # [n_dec, bs, nq, em]
        out_dec = out_dec.transpose(1, 2)

        # cls_scores
        all_cls_scores = self.cls_embed(out_dec)

        # mask_preds
        mask_embed = self.mask_embed(out_dec)
        all_mask_preds = torch.einsum('lbqc,bchw->lbqhw', mask_embed,
                                      mask_features)

        return all_cls_scores, all_mask_preds

    def forward_train(self,
                      feats,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg,
                      gt_bboxes_ignore=None):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Multi-level features from the upstream network,
                each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). #! Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor]):Each element is the ground truth
                of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            losses (dict[str, Tensor]): a dictionary of loss components
        """
        assert gt_bboxes_ignore is None  # not consider ignoring bboxes

        # forward
        all_cls_scores, all_mask_preds = self(feats)

        # preprocess ground truth
        gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks,
                                                 gt_semantic_seg)

        # loss
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks,
                           img_metas)

        return losses

    def simple_test(self, feats, img_metas, rescale=False):
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            feats (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional):  If True, return boxes in
                original image space. Default False.

        Returns:
            list[dict[str, np.array]]: semantic segmentation results
                and panoptic segmentation results for each image.
                [
                    {
                        'pan_results': np.array, # shape = [h, w]
                    },
                    ...
                ]
        """
        all_cls_scores, all_mask_preds = self(feats)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        # upsample masks
        img_shape = img_metas[0]['pad_shape'][:2]
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        results = []
        for mask_cls_result, mask_pred_result, meta in zip(
                mask_cls_results, mask_pred_results, img_metas):
            # remove padding
            img_height, img_width = meta['img_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]

            if rescale:
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(mask_pred_result.unsqueeze(1),
                                                 size=(ori_height, ori_width),
                                                 mode='bilinear',
                                                 align_corners=False)\
                    .squeeze(1)

            mask = self.post_process(mask_cls_result, mask_pred_result)
            result = {'pan_results': mask.detach().cpu().numpy()}
            results.append(result)

        return results

    def post_process(self, mask_cls, mask_pred):
        """Panoptic segmengation inference.

        This implementation is modified from
            https://github.com/facebookresearch/MaskFormer

        Args:
            mask_cls (Tensor): Classfication outputs for a image.
                shape = [num_query, cls_out_channels].
            mask_pred (Tensor): Mask outputs for a image.
                shape = [num_query, h, w].

        Returns:
            panoptic_seg (dict[str, Tensor]):
                {'pan_results': tensor of shape = (H, W) and dtype=int32},
                each element in Tensor means:
                segment_id = _cls + instance_id * INSTANCE_OFFSET.
        """
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.8)
        iou_thr = self.test_cfg.get('iou_thr', 0.8)

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > object_mask_thr)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=cur_masks.device)
        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            instance_id = 1
            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                isthing = pred_class < self.num_things_classes
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < iou_thr:
                        continue

                    if not isthing:
                        # different stuff regions of same class will be
                        # merged here, and stuff share the instance_id 0.
                        panoptic_seg[mask] = pred_class
                    else:
                        panoptic_seg[mask] = (
                            pred_class + instance_id * INSTANCE_OFFSET)
                        instance_id += 1
        return panoptic_seg
