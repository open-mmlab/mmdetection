import torch
from torch import nn
from torch.nn import functional as F

from mmdet.registry import MODELS
from .language_model import LanguageEncoder
from .transformer_blocks import (MLP, Conv2d, CrossAttentionLayer, FFNLayer,
                                 PositionEmbeddingSine, SelfAttentionLayer)
from .utils import is_lower_torch_version


def vl_similarity(image_feat, text_feat, temperature=1):
    logits = torch.matmul(image_feat, text_feat.t())
    logits = temperature.exp().clamp(max=100) * logits
    return logits


@MODELS.register_module()
class XDecoderTransformerDecoder(nn.Module):

    def __init__(
        self,
        in_channels=512,
        hidden_dim: int = 512,
        dim_proj: int = 512,
        num_queries: int = 101,
        max_token_num: int = 77,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        decoder_layers: int = 9,
        pre_norm: bool = False,
        mask_dim: int = 512,
        task: str = 'semseg',
        captioning_step: int = 50,
    ):
        super().__init__()

        # positional encoding
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # define transformer decoder here
        self.num_heads = nheads
        self.num_layers = decoder_layers
        self.max_token_num = max_token_num
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                ))

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                ))

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                ))

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim:
                self.input_proj.append(
                    Conv2d(in_channels, hidden_dim, kernel_size=1))
            else:
                self.input_proj.append(nn.Sequential())

        self.task = task

        # output FFNs
        self.lang_encoder = LanguageEncoder()

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))

        # for caption and ref-caption
        self.caping_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        self.pos_embed_caping = nn.Embedding(max_token_num, hidden_dim)
        self.captioning_step = captioning_step

        # register self_attn_mask to avoid information leakage,
        # it includes interaction between object query, class query and
        # caption query
        self_attn_mask = torch.zeros((1, num_queries + max_token_num,
                                      num_queries + max_token_num)).bool()
        # object+class query does not attend with caption query.
        self_attn_mask[:, :num_queries, num_queries:] = True
        # caption query only attend with previous token.
        self_attn_mask[:, num_queries:, num_queries:] = torch.triu(
            torch.ones((1, max_token_num, max_token_num)), diagonal=1).bool()
        # object query does not attend with class query.
        self_attn_mask[:, :num_queries - 1, num_queries - 1:num_queries] = True
        # class query does not attend with object query.
        self_attn_mask[:, num_queries - 1:num_queries, :num_queries - 1] = True
        self.register_buffer('self_attn_mask', self_attn_mask)

    def forward(self, x, mask_features, extra=None):
        if self.task == 'caption':
            return self.forward_caption(x, mask_features, extra)

        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) +
                       self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_mask = []
        predictions_class_embed = []

        if self.task == 'ref-seg':
            self_tgt_mask = self.self_attn_mask[:, :self.num_queries, :self.
                                                num_queries].repeat(
                                                    output.shape[1] *
                                                    self.num_heads, 1, 1)
            grounding_tokens = extra['grounding_tokens']
            _grounding_tokens = grounding_tokens.detach().clone()
            # initialize with negative attention at the beginning.
            pad_tgt_mask = torch.ones(
                (1, self.num_queries + (self.num_queries - 1) +
                 len(grounding_tokens), self.num_queries +
                 (self.num_queries - 1) + len(grounding_tokens)),
                device=self_tgt_mask.device).bool().repeat(
                    output.shape[1] * self.num_heads, 1, 1)
            pad_tgt_mask[:, :self.num_queries, :self.
                         num_queries] = self_tgt_mask
            # grounding tokens could attend with eatch other
            pad_tgt_mask[:, self.num_queries:, self.num_queries:] = False
            self_tgt_mask = pad_tgt_mask
            output = torch.cat((output, output[:-1]), dim=0)
            # also pad language embdding to fix embedding
            query_embed = torch.cat((query_embed, query_embed[:-1]), dim=0)
        else:
            self_tgt_mask = self.self_attn_mask[:, :self.num_queries, :self.
                                                num_queries].repeat(
                                                    output.shape[1] *
                                                    self.num_heads, 1, 1)

        results = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0])
        attn_mask = results['attn_mask']
        predictions_class_embed.append(results['class_embed'])
        predictions_mask.append(results['outputs_mask'])

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output, avg_attn = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                # here we do not apply masking on padded region
                memory_key_padding_mask=None,
                pos=pos[level_index],
                query_pos=query_embed)

            if self.task == 'ref-seg':
                output = torch.cat((output, _grounding_tokens), dim=0)
                query_embed = torch.cat((query_embed, grounding_tokens), dim=0)

            output = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=self_tgt_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed)

            output = self.transformer_ffn_layers[i](output)

            if self.task == 'ref-seg':
                _grounding_tokens = output[-len(_grounding_tokens):]
                output = output[:-len(_grounding_tokens)]
                query_embed = query_embed[:-len(_grounding_tokens)]

            results = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) %
                                                self.num_feature_levels])
            attn_mask = results['attn_mask']
            predictions_mask.append(results['outputs_mask'])
            predictions_class_embed.append(results['class_embed'])

        out = {
            'pred_masks': predictions_mask[-1],
            'pred_class_embed': predictions_class_embed[-1],
        }

        if self.task == 'ref-seg':
            mask_pred_results = []
            outputs_class = []
            for idx in range(mask_features.shape[0]):  # batch size
                pred_gmasks = out['pred_masks'][idx, self.num_queries:2 *
                                                self.num_queries - 1]
                v_emb = predictions_class_embed[-1][idx, self.num_queries:2 *
                                                    self.num_queries - 1]
                t_emb = extra['class_emb']

                t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
                v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

                temperature = self.lang_encoder.logit_scale
                out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)

                matched_id = out_prob.max(0)[1]
                mask_pred_results += [pred_gmasks[matched_id, :, :]]
                outputs_class += [out_prob[matched_id, :]]
            out['pred_masks'] = mask_pred_results
            out['pred_logits'] = outputs_class
        elif self.task == 'retrieval':
            t_emb = extra['class_emb']
            temperature = self.lang_encoder.logit_scale
            v_emb = out['pred_class_embed'][:, -1, :]
            v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
            logits = vl_similarity(v_emb, t_emb, temperature)
            out['pred_logits'] = logits
        elif self.task in ['semseg', 'instance', 'panoptic']:
            outputs_class = self.lang_encoder.compute_similarity(
                out['pred_class_embed'])
            out['pred_logits'] = outputs_class
        return out

    def forward_caption(self, x, mask_features, extra=None):
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) +
                       self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed_ = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        lang_token = extra['start_token'].repeat(bs, 1)
        pos_embed = self.pos_embed_caping.weight.unsqueeze(1).repeat(1, bs, 1)

        # prepare token embedding for evaluation
        token_embs = self.lang_encoder.lang_encoder.token_embedding.weight

        for cap_idx in range(0, self.captioning_step):
            lang_embed = self.lang_encoder.forward_language(
                (lang_token, ), with_cls_embed=False)[1].transpose(0, 1)
            # concat object query, class token and caption token.
            output = torch.cat((query_feat, lang_embed), dim=0)
            lang_embed += pos_embed
            query_embed = torch.cat((query_embed_, lang_embed), dim=0)

            # prediction heads on learnable query features
            results = self.forward_prediction_heads(
                output, mask_features, attn_mask_target_size=size_list[0])
            attn_mask = results['attn_mask']

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = torch.cat(
                    (attn_mask,
                     torch.zeros_like(attn_mask[:, :self.max_token_num, :])),
                    dim=1)
                self_tgt_mask = self.self_attn_mask.repeat(
                    output.shape[1] * self.num_heads, 1, 1)

                if 'grounding_mask' in extra:
                    bs, nq, wh = attn_mask.shape
                    assert bs == self.num_heads, 'Only support single ' \
                                                 'image referring captioning.'
                    grounding_mask = extra['grounding_mask']
                    attn_mask = attn_mask.reshape(bs, nq, size_list[i % 3][0],
                                                  size_list[i % 3][1])
                    grounding_mask = F.interpolate(
                        grounding_mask.float(),
                        size_list[i % 3],
                        mode='nearest').bool()[0, 0]
                    attn_mask[:, self.num_queries:, grounding_mask] = True
                    attn_mask = attn_mask.reshape(bs, nq, wh)

                # attention: cross-attention first
                output, avg_attn = self.transformer_cross_attention_layers[i](
                    output,
                    src[level_index],
                    memory_mask=attn_mask,
                    # here we do not apply masking on padded region
                    memory_key_padding_mask=None,
                    pos=pos[level_index],
                    query_pos=query_embed)

                output = self.transformer_self_attention_layers[i](
                    output,
                    tgt_mask=self_tgt_mask,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed)

                output = self.transformer_ffn_layers[i](output)

                results = self.forward_prediction_heads(
                    output,
                    mask_features,
                    attn_mask_target_size=size_list[(i + 1) %
                                                    self.num_feature_levels])
                attn_mask = results['attn_mask']

            pred_captions = results['outputs_caption']
            pred_captions = pred_captions @ token_embs.t()
            lang_token[:, cap_idx + 1] = pred_captions[:, cap_idx].max(-1)[1]

        texts = self.lang_encoder.tokenizer.batch_decode(
            lang_token, skip_special_tokens=False)
        texts_new = []

        for x in texts:
            x = x.split('<|endoftext|>')[0]
            x = x.replace('<|endoftext|>', '')
            x = x.replace('<|startoftext|>', '')
            x = x.strip()
            texts_new.append(x)

        out = {'pred_caption': texts_new}
        return out

    def forward_prediction_heads(self, output, mask_features,
                                 attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        if self.task == 'caption':
            outputs_caption = decoder_output[:, self.
                                             num_queries:] @ self.caping_embed

        # recompute class token output.
        norm_decoder_output = decoder_output / (
            decoder_output.norm(dim=-1, keepdim=True) + 1e-7)
        obj_token = norm_decoder_output[:, :self.num_queries - 1]
        cls_token = norm_decoder_output[:,
                                        self.num_queries - 1:self.num_queries]

        sim = (cls_token @ obj_token.transpose(1, 2)).softmax(-1)[:, 0, :,
                                                                  None]
        cls_token = (sim * decoder_output[:, :self.num_queries - 1]).sum(
            dim=1, keepdim=True)

        if self.task == 'ref-seg':
            decoder_output = torch.cat(
                (decoder_output[:, :self.num_queries - 1], cls_token,
                 decoder_output[:, self.num_queries:2 * self.num_queries - 1]),
                dim=1)
        else:
            decoder_output = torch.cat(
                (decoder_output[:, :self.num_queries - 1], cls_token), dim=1)

        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum('bqc,bchw->bqhw', mask_embed,
                                    mask_features)

        if is_lower_torch_version():
            attn_mask = F.interpolate(
                outputs_mask,
                size=attn_mask_target_size,
                mode='bicubic',
                align_corners=False)
        else:
            attn_mask = F.interpolate(
                outputs_mask,
                size=attn_mask_target_size,
                mode='bicubic',
                align_corners=False,
                antialias=True)

        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(
            1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        attn_mask[:, self.num_queries:self.num_queries + 1].fill_(False)

        if self.task == 'caption':
            results = {
                'attn_mask': attn_mask,
                'outputs_caption': outputs_caption,
            }
            return results
        else:
            class_embed = decoder_output @ self.class_embed
            results = {
                'outputs_mask': outputs_mask,
                'attn_mask': attn_mask,
                'class_embed': class_embed,
            }
            return results
