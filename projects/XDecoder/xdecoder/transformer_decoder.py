from torch import nn
import torch
from .transformer_blocks import PositionEmbeddingSine, SelfAttentionLayer, CrossAttentionLayer, FFNLayer, Conv2d, MLP
from torch.nn import functional as F
from mmdet.registry import MODELS
from .language_model import LanguageEncoder


@MODELS.register_module()
class XDecoderTransformerDecoder(nn.Module):
    def __init__(
            self,
            in_channels=512,
            hidden_dim: int = 512,
            dim_proj: int = 512,
            num_queries: int = 101,
            contxt_len: int = 77,
            nheads: int = 8,
            dim_feedforward: int = 2048,
            dec_layers: int = 9,
            pre_norm: bool = False,
            mask_dim: int = 512,
            task='semseg',
            captioning_step: int = 50,
            enforce_input_project: bool = False,
    ):
        super().__init__()
        self.mask_classification = True

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.contxt_len = contxt_len
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
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                # weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.task = task

        # output FFNs
        self.lang_encoder = LanguageEncoder()
        if self.task == 'semseg':
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        # trunc_normal_(self.class_embed, std=.02)

        # if task_switch['bbox']:
        #     self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # Caption Project and query
        # if task_switch['captioning']:
        #     self.caping_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        #     # trunc_normal_(self.caping_embed, std=.02)
        #     self.pos_embed_caping = nn.Embedding(contxt_len, hidden_dim)
        #     self.captioning_step = captioning_step

        # register self_attn_mask to avoid information leakage,
        # it includes interaction between object query, class query and caping query
        self_attn_mask = torch.zeros((1, num_queries + contxt_len, num_queries + contxt_len)).bool()
        # object+class query does not attend with caption query.
        self_attn_mask[:, :num_queries, num_queries:] = True
        # caption query only attend with previous token.
        self_attn_mask[:, num_queries:, num_queries:] = torch.triu(torch.ones((1, contxt_len, contxt_len)),
                                                                   diagonal=1).bool()
        # object query does not attend with class query.
        self_attn_mask[:, :num_queries - 1, num_queries - 1:num_queries] = True
        # class query does not attend with object query.
        self_attn_mask[:, num_queries - 1:num_queries, :num_queries - 1] = True
        self.register_buffer("self_attn_mask", self_attn_mask)

    def forward(self, x, mask_features, extra={}):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC 都是 101x512
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_bbox = []
        predictions_caption = []
        predictions_captioning = []

        self_tgt_mask = self.self_attn_mask[:, :self.num_queries, :self.num_queries].repeat(
            output.shape[1] * self.num_heads, 1, 1)  # 8,101,101

        results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        attn_mask = results["attn_mask"]
        predictions_class.append(results["outputs_class"])
        predictions_mask.append(results["outputs_mask"])
        # predictions_bbox.append(results["outputs_bbox"])
        # predictions_caption.append(results["outputs_caption"])
        # predictions_captioning.append(results["outputs_captionting"])

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output, avg_attn = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=self_tgt_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            output = self.transformer_ffn_layers[i](
                output
            )

            results = self.forward_prediction_heads(output, mask_features,
                                                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            attn_mask = results["attn_mask"]
            predictions_class.append(results["outputs_class"])
            predictions_mask.append(results["outputs_mask"])
            # predictions_bbox.append(results["outputs_bbox"])
            # predictions_caption.append(results["outputs_caption"])
            # predictions_captioning.append(results["outputs_captionting"])

        assert len(predictions_class) == self.num_layers + 1
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            # 'pred_boxes': predictions_bbox[-1],
            # 'pred_captions': predictions_caption[-1],
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, task='seg'):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        # recompute class token output.
        norm_decoder_output = decoder_output / (decoder_output.norm(dim=-1, keepdim=True) + 1e-7)
        obj_token = norm_decoder_output[:, :self.num_queries - 1]  # 101 个 query中，最后一个是 cls token，前面的100 个是 obj token
        cls_token = norm_decoder_output[:, self.num_queries - 1:self.num_queries]

        sim = (cls_token @ obj_token.transpose(1, 2)).softmax(-1)[:, 0, :, None]  # TODO include class token.
        cls_token = (sim * decoder_output[:, :self.num_queries - 1]).sum(dim=1, keepdim=True)  # 1 1 512

        decoder_output = torch.cat((decoder_output[:, :self.num_queries - 1], cls_token), dim=1)

        # compute class, mask and bbox.
        class_embed = decoder_output @ self.class_embed
        # HACK do not compute similarity if mask is not on
        outputs_class = self.lang_encoder.compute_similarity(class_embed, fake=(
                (not self.task == 'semgseg') and self.training))  # 1 101, 10

        if self.task == 'semseg':
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # 1,101,h,w

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]

            # pytorch1.7 没有 antialias 参数
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bicubic", align_corners=False,
                                      antialias=True)

            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            # NOTE: fill False for cls token (JY)
            attn_mask[:, self.num_queries:self.num_queries + 1].fill_(False)
        else:
            outputs_mask = None
            attn_mask = torch.zeros(
                (list(decoder_output.shape[:2]) + [attn_mask_target_size[0] * attn_mask_target_size[1]]),
                device=decoder_output.device).repeat(self.num_heads, 1, 1).bool()

        # outputs_bbox = [None for i in range(len(decoder_output))]
        # if self.task_switch['bbox']:
        #     outputs_bbox = self.bbox_embed(decoder_output)

        # outputs_caption = None
        # if self.task_switch['caption']:
        #     outputs_caption = class_embed

        results = {
            "outputs_class": outputs_class,
            "outputs_mask": outputs_mask,
            # "outputs_bbox": outputs_bbox,
            "attn_mask": attn_mask,
            # "outputs_caption": outputs_caption,
        }
        return results
