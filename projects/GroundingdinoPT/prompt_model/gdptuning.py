# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import OptConfigType


@MODELS.register_module()
class GroundingDINOPTuning(GroundingDINO):
    """Implementation of prompt tuning in Grounding DINO."""

    def __init__(self, prompt_cfg: OptConfigType, *args, **kwargs) -> None:
        self.class_num = prompt_cfg['class_num']
        self.prompt_length = prompt_cfg['prompt_length']
        self.prompt_train = True
        super().__init__(*args, **kwargs)

        self.learning_prompts = nn.ModuleList([
            nn.Embedding(
                self.prompt_length,
                self.language_model.language_backbone.body.language_dim)
            for _ in range(self.class_num)
        ])
        self._freeze_stages()

    def _freeze_stages(self):
        self.backbone.eval()
        self.bbox_head.eval()
        self.decoder.eval()
        self.encoder.eval()
        self.language_model.eval()
        self.memory_trans_norm.eval()
        self.neck.eval()
        for name, param in self.named_parameters():
            if 'learning_prompts' in name:
                continue
            elif 'label_embedding' in name:
                continue
            else:
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(GroundingDINOPTuning, self).train(mode)
        self._freeze_stages()

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        self.prompt_train = True
        text_prompts = []
        for data_samples in batch_data_samples:
            word = 'a ' * self.prompt_length
            pseudo_words = [word.strip() for _ in range(self.class_num)]
            text_prompts.append(pseudo_words)

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        new_text_prompts = []
        positive_maps = []

        tokenized, caption_string, tokens_positive, _ = \
            self.get_tokens_and_prompts(
                text_prompts[0], True)
        new_text_prompts = [caption_string] * len(batch_inputs)
        for gt_label in gt_labels:
            new_tokens_positive = [
                tokens_positive[label] for label in gt_label
            ]
            _, positive_map = self.get_positive_map(tokenized,
                                                    new_tokens_positive)
            positive_maps.append(positive_map)

        text_dict = self.language_model(new_text_prompts)

        #   insert learnable prompt
        insert_map, _ = self.get_positive_map(tokenized, tokens_positive)
        for i in range(self.class_num):
            cur = self.learning_prompts[i].weight.repeat(
                text_dict['embedded'].shape[0], 1, 1)
            text_dict['embedded'][:,
                                  insert_map[i + 1][0]:insert_map[i + 1][-1] +
                                  1, :] = cur

        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        self.prompt_train = False
        text_prompts = []
        label_list = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text = data_samples.text
            if isinstance(text, str):
                text = text.lower().strip().strip('.')
                text = [c.strip() for c in text.split('.')] if text else []
            text = list(text)

            if hasattr(data_samples, 'prompt_pth'):
                visual_query, real_name, pseudo_words = \
                    self.predict_add_prompt(data_samples)
                real_name.extend(text)
                pseudo_words.extend(text)
                label_list.append(real_name)
                text_prompts.append(pseudo_words)
            else:
                word = 'a ' * self.prompt_length
                pseudo_words = [word.strip() for _ in range(self.class_num)]
                text_prompts.append(pseudo_words)
                label_list.append(text)
            enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        _positive_maps_and_prompts = [
            self.get_tokens_positive_and_prompts(text_prompt, custom_entities,
                                                 enhanced_text_prompt,
                                                 tokens_positive)
            for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                text_prompts, enhanced_text_prompts, tokens_positives)
        ]
        token_positive_maps, text_prompts, _, _ = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        # extract text feats
        text_dict = self.language_model(list(text_prompts))

        entities = label_list
        if hasattr(data_samples, 'prompt_pth'):
            positive_map = token_positive_maps[0]
            for i in range(len(visual_query)):
                cur = visual_query[i].repeat(text_dict['embedded'].shape[0], 1,
                                             1)
                text_dict['embedded'][:,
                                      positive_map[i +
                                                   1][0]:positive_map[i +
                                                                      1][-1] +
                                      1, :] = cur
        else:
            positive_map = token_positive_maps[0]
            for i in range(self.class_num):
                cur = self.learning_prompts[i].weight.repeat(
                    text_dict['embedded'].shape[0], 1, 1)
                text_dict['embedded'][:,
                                      positive_map[i +
                                                   1][0]:positive_map[i +
                                                                      1][-1] +
                                      1, :] = cur

        # text feature map layer
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        is_rec_tasks = []
        for i, data_samples in enumerate(batch_data_samples):
            if token_positive_maps[i] is not None:
                is_rec_tasks.append(False)
            else:
                is_rec_tasks.append(True)
            data_samples.token_positive_map = token_positive_maps[i]

        head_inputs_dict = self.forward_transformer(visual_feats, text_dict,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples

    def predict_add_prompt(self, data_samples):
        embedding_path = data_samples.prompt_pth
        real_name = []
        pseudo_words = []
        visual_query = []
        for a_embedding in embedding_path:
            if isinstance(a_embedding, str):
                root_file = osp.abspath(osp.dirname(osp.dirname(__file__)))
                embedding_abs_path = osp.normpath(
                    osp.join(root_file, osp.expanduser(a_embedding)))
                embedding = torch.load(embedding_abs_path)
            else:
                import io
                buffer = io.BytesIO(a_embedding)
                embedding = torch.load(buffer)
            for k, v in embedding.items():
                visual_query.append(v['embeddning'])
                real_name.append(k)
                word = 'a ' * v['embeddning'].shape[0]
                pseudo_words.append(word.strip())
        return visual_query, real_name, pseudo_words

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training or self.prompt_train:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training or self.prompt_train else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict
