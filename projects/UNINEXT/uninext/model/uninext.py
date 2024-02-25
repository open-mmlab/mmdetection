import copy
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn.init import normal_

from mmdet.models.detectors.deformable_detr import (
    DeformableDETR, MultiScaleDeformableAttention)
from mmdet.models.layers import SinePositionalEncoding
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, TrackSampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import OptConfigType
from .layers import (BertEncoder, BertTokenizer, TextCdnQueryGenerator,
                     UninextTransformerDecoder)
from .utils import agg_lang_feat
from .vifusion import VLTransformerEncoder


@MODELS.register_module()
class UNINEXT_VID(DeformableDETR):

    def __init__(self,
                 *args,
                 tokenizer_cfg: OptConfigType = None,
                 text_encoder_cfg: OptConfigType = None,
                 dn_cfg: OptConfigType = None,
                 tracker: Optional[dict] = dict(
                     type='IDOLTracker',
                     init_score_thr=0.2,
                     obj_score_thr=0.1,
                     nms_thr_pre=0.5,
                     nms_thr_post=0.05,
                     addnew_score_thr=0.2,
                     memo_tracklet_frames=10,
                     memo_momentum=0.8,
                     long_match=True,
                     frame_weight=True,
                     temporal_weight=True,
                     memory_len=3,
                     match_metric='bisoftmax'),
                 **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.tracker = MODELS.build(tracker)

        # dn
        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            dn_cfg['num_classes'] = 80  # not use in TextCdnQueryGenerator
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
        self.dn_query_generator = TextCdnQueryGenerator(**dn_cfg)

        # text
        self.tokenizer = BertTokenizer(**tokenizer_cfg)
        self.text_encoder = BertEncoder(**text_encoder_cfg)
        assert 'frozen_parameters' in text_encoder_cfg
        if text_encoder_cfg.get('frozen_parameters', True):
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""

        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.vl_encoder = VLTransformerEncoder(**self.encoder)
        self.decoder = UninextTransformerDecoder(**self.decoder)
        self.embed_dims = self.vl_encoder.embed_dims

        self.land_fims = self.bbox_head.language_dims
        self.resizer = nn.Sequential(
            nn.Linear(self.land_fims, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims, eps=1e-12), nn.Dropout(0.0))
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        # for coder in self.encoder, self.decoder:
        for coder in self.encoder.vision_layers, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        normal_(self.level_embed)

    def forward_text(self, captions, device):

        # captions = ''
        assert isinstance(captions[0], str)
        tokenizer_input = self.tokenizer(captions, device)
        language_dict_features = self.text_encoder(tokenizer_input)
        # dict with keys: ['aggregate', 'embedded', 'masks', 'hidden']
        return language_dict_features

    def loss(self,
             batch_inputs: Tensor,
             data_samples: TrackSampleList,
             task: str = 'detection') -> Union[dict, tuple]:

        assert batch_inputs.dim(
        ) == 5, 'The img must be 5D Tensor (B, T, C, H, W).'
        assert batch_inputs.size(1) == 2, \
            'UNINEXT can only have 1 key frame and 1 reference frame.'

        # split the data_samples into two aspects: key frames and reference
        # frames
        key_data_samples, ref_data_samples = [], []
        key_frame_inds, ref_frame_inds = [], []
        key_captions, ref_captions = [], []

        # [batch]
        for track_data_sample in data_samples:
            key_frame_inds.append(track_data_sample.key_frames_inds[0])
            ref_frame_inds.append(track_data_sample.ref_frames_inds[0])
            key_data_sample = track_data_sample.get_key_frames()[0]
            ref_data_sample = track_data_sample.get_ref_frames()[0]
            key_captions.append(key_data_sample.expressions)
            ref_captions.append(ref_captions.expressions)
            key_data_sample.gt_instances.bbox = bbox_xyxy_to_cxcywh(
                key_data_sample.gt_instances.bbox)
            ref_data_sample.gt_instances.bbox = bbox_xyxy_to_cxcywh(
                ref_data_sample.gt_instances.bbox)
            key_valid = (key_data_sample.gt_instances.instances_ids) != -1
            if False in key_valid:
                key_data_sample.gt_instances = key_data_sample.gt_instances[
                    key_valid]
                ref_data_sample.gt_instances = ref_data_sample.gt_instances[
                    key_valid]
            key_data_samples.append(key_data_sample)
            ref_data_samples.append(ref_data_sample)

        key_frame_inds = torch.tensor(key_frame_inds, dtype=torch.int64)
        ref_frame_inds = torch.tensor(ref_frame_inds, dtype=torch.int64)
        batch_inds = torch.arange(len(batch_inputs))
        key_imgs = batch_inputs[batch_inds, key_frame_inds].contiguous()
        ref_imgs = batch_inputs[batch_inds, ref_frame_inds].contiguous()

        key_img_feats = self.extract_feat(key_imgs)
        ref_img_feats = self.extract_feat(ref_imgs)
        language_dict_features_key = self.forward_text(
            key_captions, device=batch_inputs.device)
        language_dict_features_ref = self.forward_text(
            ref_captions, device=batch_inputs.device)

        key_result = self.forward_transformer(
            key_img_feats,
            language_dict_features_key,
            key_data_samples,
            key_image=True)
        ref_result = self.forward_transformer(
            ref_img_feats,
            language_dict_features_ref,
            ref_data_samples,
            key_image=False)

        result, loss_dict = self.track_head.loss(key_result, ref_result,
                                                 key_data_samples,
                                                 ref_data_samples, task)
        # weight_dict = self.criterion.weight_dict
        # for k in loss_dict.keys():
        #     if k in weight_dict:
        #         loss_dict[k] *= weight_dict[k]
        # losses = sum(loss_dict.values())
        return loss_dict

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            language_dict_features: Dict,
                            batch_data_samples: OptSampleList = None,
                            is_key_image: bool = True) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        # use in dn
        lang_feat_pool = agg_lang_feat(
            language_dict_features['hidden'],
            language_dict_features['masks'],
            pool_type='average')  # (bs, 768)
        label_enc = self.resizer(lang_feat_pool)  # (bs, 256)

        encoder_outputs_dict, src_info_dict = self.forward_vlfusion_encoder(
            language_dict_features, **encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict,
            label_enc=label_enc,
            is_key_image=is_key_image,
            batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        head_inputs_dict.update({'src_info_dict': src_info_dict})
        return head_inputs_dict

    def forward_vlfusion_encoder(self, language_dict_features: Tensor,
                                 feat: Tensor, feat_mask: Tensor,
                                 feat_pos: Tensor, spatial_shapes: Tensor,
                                 level_start_index: Tensor,
                                 valid_ratios: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """

        vl_feats_dict = self.vl_encoder(
            language_dict_features=language_dict_features,
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        memory, language_dict_features = vl_feats_dict[
            'visual'], vl_feats_dict['lang']
        encoder_outputs_dict = dict(
            memory=memory,
            language_dict_features=language_dict_features,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
        )
        src_info_dict = {
            'src': memory.detach(),
            'src_spatial_shapes': spatial_shapes,
            'src_level_start_index': level_start_index,
            'src_valid_ratios': valid_ratios
        }
        return encoder_outputs_dict, src_info_dict

    def pre_decoder(
        self,
        memory: Tensor,
        language_dict_features: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        label_enc: Tensor,
        is_key_image: bool = True,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs = memory.size(0)

        lang_feat_pool = agg_lang_feat(
            language_dict_features['hidden'],
            language_dict_features['masks'])  # (bs, 768)
        ref_feat = self.resizer(lang_feat_pool)  # (bs, 256)
        ref_feat = ref_feat.unsqueeze(1)  # (bs, 1, 256)

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        topk_proposals = torch.topk(
            enc_outputs_class[..., 0], k=self.num_queries, dim=1)[1]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4))

        # mixed query selection
        query = self.query_embedding.weight[None].repeat(bs, 1, 1)
        if self.training:
            if is_key_image:
                dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                    self.dn_query_generator(batch_data_samples, label_enc)
                query = torch.cat([dn_label_query, query], dim=1)
                reference_points = torch.cat(
                    [dn_bbox_query, topk_coords_unact], dim=1)
            else:
                reference_points = topk_coords_unact
                dn_mask, dn_meta = None, None
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        if ref_feat is not None:
            query = query + 0.0 * ref_feat  # both use the original tgt

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)

        if self.training:
            head_inputs_dict = dict(
                memory=memory,
                enc_outputs_class=enc_outputs_class,
                enc_outputs_coord=enc_outputs_coord_unact,
                language_dict_features=language_dict_features,
                dn_meta=dn_meta)
        else:
            head_inputs_dict = dict(
                memory=memory, language_dict_features=language_dict_features)
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        inter_states, inter_references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches)

        decoder_outputs_dict = dict(
            hidden_states=inter_states, inter_references=inter_references)
        return decoder_outputs_dict

    def predict(self,
                inputs: Dict[str, Tensor],
                data_samples: TrackSampleList,
                rescale: bool = True) -> TrackSampleList:
        """Predict results from a video and data samples with post-processing.

        Args:
            inputs (Tensor): of shape (N, T, C, H, W) encoding
                input images. The N denotes batch size.
                The T denotes the number of frames in a video.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `video_data_samples`.
        Returns:
            TrackSampleList: Tracking results of the inputs.
        """
        assert inputs.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert inputs.size(0) == 1, \
            'UNINEXT inference only support ' \
            '1 batch size per gpu for now.'

        assert len(data_samples) == 1, \
            'UNINEXT inference only support 1 batch size per gpu for now.'

        track_data_sample = data_samples[0]
        # task = track_data_sample.task
        video_len = len(track_data_sample)
        self.tracker.reset()

        captions = [
            track_data_sample[frame_id].expressions
            for frame_id in range(video_len)
        ]
        assert len(set(captions)) == 1
        language_dict_features = self.forward_text(
            captions[0:1], device=inputs.device)

        for frame_id in range(video_len):
            # if frame_id == 21:
            #     print("hhh")
            img_data_sample = track_data_sample[frame_id]
            img_data_sample.set_metainfo({'frame_id': frame_id})
            single_img = inputs[:, frame_id].contiguous()  # (B, C, H, W)
            positive_map_label_to_token = \
                img_data_sample.positive_map_label_to_token
            num_classes = len(positive_map_label_to_token)
            language_dict_features_cur = copy.deepcopy(language_dict_features)

            img_feats = self.extract_feat(single_img)
            result = self.forward_transformer(img_feats,
                                              language_dict_features_cur,
                                              [img_data_sample])
            pred_det_ins_list = self.bbox_head.predict(
                **result,
                positive_map_label_to_token=positive_map_label_to_token,
                num_classes=num_classes,
                batch_data_samples=[img_data_sample],
                rescale=rescale)
            img_data_sample.pred_instances = pred_det_ins_list[0]
            pred_track_instances = self.tracker.track(
                data_sample=img_data_sample, rescale=rescale)
            img_data_sample.pred_track_instances = pred_track_instances

        return [track_data_sample]

    def predict_sot(self,
                    inputs: Dict[str, Tensor],
                    data_samples: TrackSampleList,
                    rescale: bool = True) -> TrackSampleList:
        """Predict results from a video and data samples with post-processing.

        Args:
            inputs (Tensor): of shape (N, T, C, H, W) encoding
                input images. The N denotes batch size.
                The T denotes the number of frames in a video.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `video_data_samples`.
        Returns:
            TrackSampleList: Tracking results of the inputs.
        """
        assert inputs.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert inputs.size(0) == 1, \
            'UNINEXT inference only support ' \
            '1 batch size per gpu for now.'

        assert len(data_samples) == 1, \
            'UNINEXT inference only support 1 batch size per gpu for now.'

        track_data_sample = data_samples[0]
        # task = track_data_sample.task
        video_len = len(track_data_sample)
        self.tracker.reset()

        captions = [
            track_data_sample[frame_id].expressions
            for frame_id in range(video_len)
        ]
        assert len(set(captions)) == 1
        language_dict_features = self.forward_text(
            captions[0:1], device=inputs.device)

        for frame_id in range(video_len):
            img_data_sample = track_data_sample[frame_id]
            img_data_sample.set_metainfo({'frame_id': frame_id})
            single_img = inputs[:, frame_id].contiguous()  # (B, C, H, W)
            positive_map_label_to_token = \
                img_data_sample.positive_map_label_to_token
            num_classes = len(positive_map_label_to_token)
            language_dict_features_cur = copy.deepcopy(language_dict_features)

            img_feats = self.extract_feat(single_img)
            result = self.forward_transformer(img_feats,
                                              language_dict_features_cur,
                                              [img_data_sample])
            pred_det_ins_list = self.bbox_head.predict(
                **result,
                positive_map_label_to_token=positive_map_label_to_token,
                num_classes=num_classes,
                batch_data_samples=[img_data_sample],
                rescale=rescale)
            img_data_sample.pred_instances = pred_det_ins_list[0]
            pred_track_instances = self.tracker.track(
                data_sample=img_data_sample, rescale=rescale)
            img_data_sample.pred_track_instances = pred_track_instances

        return [track_data_sample]
