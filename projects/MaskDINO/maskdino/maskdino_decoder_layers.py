# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional

from mmengine.model import caffe2_xavier_init
import torch
from mmcv.cnn import ConvModule
from mmcv.ops import MultiScaleDeformableAttention
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import functional as F

from mmdet.models.layers import MLP, coordinate_to_encoding, inverse_sigmoid
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh


def setup_seed(seed):
    import random

    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks.

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


class MaskDINODecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        mask_dim: int,
        enforce_input_project: bool,
        two_stage: bool,
        dn: str,
        noise_scale: float,
        dn_num: int,
        initialize_box_type: bool,
        initial_pred: bool,
        learn_tgt: bool,
        total_num_feature_levels: int = 4,
        dropout: float = 0.0,
        activation: str = 'relu',
        nhead: int = 8,
        dec_n_points: int = 4,
        mask_classification=True,
        return_intermediate_dec: bool = True,
        query_dim: int = 4,
        dec_layer_share: bool = False,
        semantic_ce_loss: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            d_model: transformer dimension
            dropout: dropout rate
            activation: activation function
            nhead: num heads in multi-head attention
            dec_n_points: number of sampling points in decoder
            return_intermediate_dec: return the intermediate results of decoder
            query_dim: 4 -> (x, y, w, h)
            dec_layer_share: whether to share each decoder layer
            semantic_ce_loss: use ce loss for semantic segmentation
        """
        super().__init__()

        assert mask_classification, 'Only support mask classification model'
        self.mask_classification = mask_classification
        self.num_feature_levels = total_num_feature_levels
        self.initial_pred = initial_pred

        # define Transformer decoder here
        self.dn = dn
        self.learn_tgt = learn_tgt
        self.noise_scale = noise_scale
        self.dn_num = dn_num
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.two_stage = two_stage
        self.initialize_box_type = initialize_box_type
        self.total_num_feature_levels = total_num_feature_levels

        self.num_queries = num_queries
        self.semantic_ce_loss = semantic_ce_loss
        # learnable query features
        if not two_stage or self.learn_tgt:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
        if not two_stage and initialize_box_type == 'no':
            self.query_embed = nn.Embedding(num_queries, 4)
        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(
                    ConvModule(
                        in_channels, hidden_dim, kernel_size=1, act_cfg=None))
                # weight_init.c2_xavier_fill(self.input_proj[-1])
                caffe2_xavier_init(self.input_proj[-1].conv)
            else:
                self.input_proj.append(nn.Sequential())
        self.num_classes = num_classes
        # output FFNs
        assert self.mask_classification, 'why not class embedding?'
        if self.mask_classification:
            if self.semantic_ce_loss:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            else:
                self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.label_enc = nn.Embedding(num_classes, hidden_dim)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # init decoder
        # self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, dim_feedforward, dropout, activation,
            self.num_feature_levels, nhead, dec_n_points)
        self.decoder = TransformerDecoder(
            decoder_layer,
            self.num_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=hidden_dim,
            query_dim=query_dim,
            num_feature_levels=self.num_feature_levels,
            dec_layer_share=dec_layer_share)
        self.hidden_dim = hidden_dim
        # self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [
            _bbox_embed for i in range(self.num_layers)
        ]  # TODO: Notice: share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)

    def prepare_for_dn(self, targets, tgt, refpoint_emb, batch_size):
        """modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-
        DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py for more
        details.

        :param dn_args: scalar, noise_scale
        :param tgt: original tgt (content) in the matching part
        :param refpoint_emb: positional anchor queries in the matching part
        :param batch_size: bs
        """
        if self.training:
            scalar, noise_scale = self.dn_num, self.noise_scale

            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            if max(known_num) > 0:
                scalar = scalar // (int(max(known_num)))
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_bbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_bbox, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([
                torch.full_like(t['labels'].long(), i)
                for i, t in enumerate(targets)
            ])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_bboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            # setup_seed(20)
            # noise on the label
            if noise_scale > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(
                    -1)  # half of bbox prob
                new_label = torch.randint_like(
                    chosen_indice, 0,
                    self.num_classes)  # randomly put a new one here
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
            if noise_scale > 0:
                diff = torch.zeros_like(known_bbox_expand)
                diff[:, :2] = known_bbox_expand[:, 2:] / 2
                diff[:, 2:] = known_bbox_expand[:, 2:]
                known_bbox_expand += torch.mul(
                    (torch.rand_like(known_bbox_expand) * 2 - 1.0),
                    diff).cuda() * noise_scale
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

            m = known_labels_expaned.long().to('cuda')
            input_label_embed = self.label_enc(m)
            input_bbox_embed = inverse_sigmoid(known_bbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)

            padding_label = torch.zeros(pad_size, self.hidden_dim).cuda()
            padding_bbox = torch.zeros(pad_size, 4).cuda()

            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt],
                                              dim=0).repeat(batch_size, 1, 1)
                input_query_bbox = torch.cat([padding_bbox, refpoint_emb],
                                             dim=0).repeat(batch_size, 1, 1)
            else:
                input_query_label = padding_label.repeat(batch_size, 1, 1)
                input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

            # map
            map_known_indice = torch.tensor([]).to('cuda')
            if len(known_num):
                map_known_indice = torch.cat([
                    torch.tensor(range(num)) for num in known_num
                ])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([
                    map_known_indice + single_pad * i for i in range(scalar)
                ]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(),
                                   map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(),
                                  map_known_indice)] = input_bbox_embed

            tgt_size = pad_size + self.num_queries
            attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1),
                              single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad *
                              (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1),
                              single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad *
                              (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        else:
            if not refpoint_emb is None:
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
            else:
                input_query_label = None
                input_query_bbox = None
            attn_mask = None
            mask_dict = None

        # 100*batch*256
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox

        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def dn_post_process(self, outputs_class, outputs_coord, mask_dict,
                        outputs_mask):
        """post process of dn after output from the transformer put the dn part
        in the mask_dict."""
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
        out = {
            'pred_logits': output_known_class[-1],
            'pred_boxes': output_known_coord[-1],
            'pred_masks': output_known_mask[-1]
        }

        out['aux_outputs'] = self._set_aux_loss(output_known_class,
                                                output_known_mask,
                                                output_known_coord)
        mask_dict['output_known_lbs_bboxes'] = out
        return outputs_class, outputs_coord, outputs_mask

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def pred_box(self, reference, hs, ref0=None):
        """
        :param reference: reference box coordinates from each decoder layer
        :param hs: content
        :param ref0: whether there are prediction from the first layer
        """
        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(
                layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list

    def forward(self, x, mask_features, masks, targets=None):
        """
        :param x: input, a list of multi-scale feature
        :param mask_features: is the per-pixel embeddings with resolution 1/4 of the original image,
        obtained by fusing backbone encoder encoded features. This is used to produce binary masks.
        :param masks: mask in the original image
        :param targets: used for denoising training
        """
        assert len(x) == self.num_feature_levels
        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [
                torch.zeros((src.size(0), src.size(2), src.size(3)),
                            device=src.device,
                            dtype=torch.bool) for src in x
            ]
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            bs, c, h, w = x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](
                x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_mask = []
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                src_flatten, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(
                self.enc_output(output_memory))
            enc_outputs_class_unselected = self.class_embed(output_memory)
            # enc_outputs_coord_unselected = self._bbox_embed(
            #     output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
            enc_outputs_coord_unselected = self.bbox_embed[-1](
                output_memory
            ) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries
            topk_proposals = torch.topk(
                enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
            refpoint_embed = refpoint_embed_undetach.detach()

            tgt_undetach = torch.gather(output_memory, 1,
                                        topk_proposals.unsqueeze(-1).repeat(
                                            1, 1,
                                            self.hidden_dim))  # unsigmoid

            outputs_class, outputs_mask = self.forward_prediction_heads(
                tgt_undetach.transpose(0, 1), mask_features)
            tgt = tgt_undetach.detach()
            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            interm_outputs = dict()
            interm_outputs['pred_logits'] = outputs_class
            interm_outputs['pred_boxes'] = refpoint_embed_undetach.sigmoid()
            interm_outputs['pred_masks'] = outputs_mask

            if self.initialize_box_type != 'no':
                # convert masks into boxes to better initialize box in the decoder
                assert self.initial_pred
                flaten_mask = outputs_mask.detach().flatten(0, 1)
                h, w = outputs_mask.shape[-2:]
                if self.initialize_box_type == 'bitmask':  # slower, but more accurate
                    raise NotImplementedError()  # TODO: learn to write this
                    # refpoint_embed = BitMasks(flaten_mask > 0).get_bounding_boxes().tensor.cuda()  # TODO: make a dummy BitMask?
                elif self.initialize_box_type == 'mask2box':  # faster conversion
                    refpoint_embed = masks_to_boxes(flaten_mask > 0)
                else:
                    assert NotImplementedError
                refpoint_embed = bbox_xyxy_to_cxcywh(
                    refpoint_embed) / torch.as_tensor(
                        [w, h, w, h], dtype=torch.float).cuda()
                refpoint_embed = refpoint_embed.reshape(
                    outputs_mask.shape[0], outputs_mask.shape[1], 4)
                refpoint_embed = inverse_sigmoid(refpoint_embed)
        elif not self.two_stage:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)

        tgt_mask = None
        mask_dict = None
        if self.dn != 'no' and self.training:
            assert targets is not None
            input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                self.prepare_for_dn(targets, None, None, x[0].shape[0])
            if mask_dict is not None:
                tgt = torch.cat([input_query_label, tgt], dim=1)

        # direct prediction from the matching and denoising part in the beginning
        if self.initial_pred:
            outputs_class, outputs_mask = self.forward_prediction_heads(
                tgt.transpose(0, 1), mask_features, self.training)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
        if self.dn != 'no' and self.training and mask_dict is not None:
            refpoint_embed = torch.cat([input_query_bbox, refpoint_embed],
                                       dim=1)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask,
            bbox_embed=self.bbox_embed)
        for i, output in enumerate(hs):
            outputs_class, outputs_mask = self.forward_prediction_heads(
                output.transpose(0, 1), mask_features, self.training
                or (i == len(hs) - 1))
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        # iteratively box prediction
        if self.initial_pred:
            out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
            assert len(predictions_class) == self.num_layers + 1
        else:
            out_boxes = self.pred_box(references, hs)
        if mask_dict is not None:
            predictions_mask = torch.stack(predictions_mask)
            predictions_class = torch.stack(predictions_class)
            predictions_class, out_boxes, predictions_mask = \
                self.dn_post_process(predictions_class, out_boxes, mask_dict, predictions_mask)
            predictions_class, predictions_mask = list(
                predictions_class), list(predictions_mask)
        elif self.training:  # this is to insure self.label_enc participate in the model
            predictions_class[-1] += 0.0 * self.label_enc.weight.sum()

        out = {
            'pred_logits':
            predictions_class[-1],
            'pred_masks':
            predictions_mask[-1],
            'pred_boxes':
            out_boxes[-1],
            'aux_outputs':
            self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask, out_boxes)
        }
        if self.two_stage:
            out['interm_outputs'] = interm_outputs
        return out, mask_dict

    def forward_prediction_heads(self, output, mask_features, pred_mask=True):
        # decoder_output = self.decoder_norm(output)  # TODO: again ???
        decoder_output = self.decoder.norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        outputs_mask = None
        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum('bqc,bchw->bqhw', mask_embed,
                                        mask_features)

        return outputs_class, outputs_mask

    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_boxes=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        if out_boxes is None:
            return [{
                'pred_logits': a,
                'pred_masks': b
            } for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            return [{
                'pred_logits': a,
                'pred_masks': b,
                'pred_boxes': c
            } for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1],
                                 out_boxes[:-1])]


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        modulate_hw_attn=True,
        num_feature_levels=1,
        deformable_decoder=True,
        decoder_query_perturber=None,
        dec_layer_number=None,  # number of queries each layer in decoder
        rm_dec_query_scale=True,
        dec_layer_share=False,
        dec_layer_dropout_prob=None,
    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(
                decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, 'support return_intermediate only'
        self.query_dim = query_dim
        assert query_dim in [
            2, 4
        ], 'query_dim should be 2/4 but {}'.format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model,
                                  2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        # self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers
            # assert dec_layer_number[0] ==

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def forward(
            self,
            tgt,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
            # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
            spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
            valid_ratios: Optional[Tensor] = None,
            bbox_embed: Optional[nn.Module] = None):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(
                    reference_points)

            reference_points_input = reference_points[:, :, None] \
                                     * torch.cat([valid_ratios, valid_ratios], -1)[None, :]  # nq, bs, nlevel, 4
            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])  # nq, bs, 256*2

            raw_query_pos = self.ref_point_head(
                query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(
                output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask)

            # iter update
            if bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return [[itm_out.transpose(0, 1) for itm_out in intermediate],
                [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]]


class DeformableTransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation='relu',
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_deformable_box_attn=False,
        key_aware_type=None,
    ):
        super().__init__()

        # cross attention
        if use_deformable_box_attn:
            raise NotImplementedError
        else:
            self.cross_attn = MultiScaleDeformableAttention(
                embed_dims=d_model,
                num_levels=n_levels,
                num_heads=n_heads,
                num_points=n_points,
                dropout=dropout)
        # self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    @autocast(enabled=False)
    def forward(
            self,
            # for tgt
            tgt: Optional[Tensor],  # nq, bs, d_model
            tgt_query_pos: Optional[
                Tensor] = None,  # pos for query. MLP(Sine(pos))
            tgt_query_sine_embed: Optional[
                Tensor] = None,  # pos for query. Sine(pos)
            tgt_key_padding_mask: Optional[Tensor] = None,
            tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4

            # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
            memory_key_padding_mask: Optional[Tensor] = None,
            memory_level_start_index: Optional[Tensor] = None,  # num_levels
            memory_spatial_shapes: Optional[
                Tensor] = None,  # bs, num_levels, 2
            memory_pos: Optional[Tensor] = None,  # pos for memory

            # sa
        self_attn_mask: Optional[
            Tensor] = None,  # mask used for self-attention
            cross_attn_mask: Optional[
                Tensor] = None,  # mask used for cross-attention
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        # self attention
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # cross attention
        if self.key_aware_type is not None:
            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError('Unknown key_aware_type: {}'.format(
                    self.key_aware_type))
        # tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
        #                        tgt_reference_points.transpose(0, 1).contiguous(),
        #                        memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index,
        #                        memory_key_padding_mask).transpose(0, 1)
        tgt = self.cross_attn(
            query=tgt,
            query_pos=tgt_query_pos,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index)
        # tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


def gen_encoder_output_proposals(memory: Tensor, memory_padding_mask: Tensor,
                                 spatial_shapes: Tensor):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(
            N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(
                0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(
                0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_W.unsqueeze(-1),
                           valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) &
                              (output_proposals < 0.99)).all(
                                  -1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(
        memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid,
                                                    float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(
        memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid,
                                              float(0))
    return output_memory, output_proposals


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    if activation == 'prelu':
        return nn.PReLU()
    if activation == 'selu':
        return F.selu
    raise RuntimeError(F'activation should be relu/gelu, not {activation}.')


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
