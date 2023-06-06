import os
from collections import OrderedDict

import torch
from mmcv.cnn.bricks import DropPath
from torch import nn
from transformers import CLIPTokenizer

from .utils import get_prompt_templates

# modified from https://github.com/microsoft/X-Decoder/blob/main/xdecoder/language/vlpencoder.py # noqa


class LanguageEncoder(nn.Module):

    def __init__(
        self,
        tokenizer='openai/clip-vit-base-patch32',
        dim_lang=512,
        dim_projection=512,
    ):
        super().__init__()

        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_special_tokens(
            {'cls_token': self.tokenizer.eos_token})

        max_token_num = self.tokenizer.model_max_length
        self.lang_encoder = Transformer(max_token_num,
                                        self.tokenizer.vocab_size, dim_lang)

        self.lang_proj = nn.Parameter(torch.empty(dim_lang, dim_projection))
        self.max_token_num = max_token_num
        self.logit_scale = nn.Parameter(torch.ones([]))

    @torch.no_grad()
    def get_mean_embeds(self, class_names, name='default'):

        def extract_mean_emb(txts):
            tokens = self.tokenizer(
                txts,
                padding='max_length',
                truncation=True,
                max_length=self.max_token_num,
                return_tensors='pt')
            clss_embedding, _ = self.forward_language(
                (tokens['input_ids'].cuda(), tokens['attention_mask'].cuda()),
                norm=True,
                with_token_embed=False)
            clss_embedding = clss_embedding.mean(dim=0)
            clss_embedding /= clss_embedding.norm()
            return clss_embedding

        templates = get_prompt_templates()

        clss_embeddings = []
        for clss in class_names:
            txts = [
                template.format(
                    clss.replace('-other',
                                 '').replace('-merged',
                                             '').replace('-stuff', ''))
                for template in templates
            ]
            clss_embeddings.append(extract_mean_emb(txts))

        text_emb = torch.stack(clss_embeddings, dim=0)
        setattr(self, '{}_text_embeddings'.format(name), text_emb)

    def get_text_embeds(self, txts, name='grounding', norm=False):
        tokens = self.tokenizer(
            txts,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_num,
            return_tensors='pt')
        tokens = {key: value.cuda() for key, value in tokens.items()}
        class_emb, token_emb = self.forward_language(
            (tokens['input_ids'], tokens['attention_mask']), norm=norm)
        ret = {
            'tokens': tokens,
            'token_emb': token_emb,
            'class_emb': class_emb,
        }
        setattr(self, '{}_token_embeddings'.format(name), ret)
        return ret

    def get_sot_token(self, device):
        # 49406: CLIP SOT token <|startoftext|>
        # 77: CLIP context_length
        return torch.tensor([[49406] * 77], device=device)

    def compute_similarity(self, v_emb, name='default'):
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
        t_emb = getattr(self, '{}_text_embeddings'.format(name))
        output = self.logit_scale.exp() * v_emb @ t_emb.unsqueeze(0).transpose(
            1, 2)
        return output

    def forward_language(self,
                         texts,
                         norm=False,
                         with_token_embed=True,
                         with_cls_embed=True):
        x = self.lang_encoder(*texts)
        hidden_x = x['last_hidden_state']

        class_embed = None
        if with_cls_embed:
            class_embed = hidden_x[torch.arange(hidden_x.size(0)),
                                   texts[0].argmax(dim=-1)]

            class_embed = class_embed @ self.lang_proj
            if norm:
                class_embed = class_embed / (
                    class_embed.norm(dim=-1, keepdim=True) + 1e-7)

        hidden_embed = None
        if with_token_embed:
            hidden_embed = hidden_x @ self.lang_proj
            if norm:
                hidden_embed = hidden_embed / (
                    hidden_embed.norm(dim=-1, keepdim=True) + 1e-7)

        return class_embed, hidden_embed


class Transformer(nn.Module):

    def __init__(self,
                 context_length,
                 vocab_size,
                 width,
                 layers: int = 12,
                 heads: int = 8,
                 drop_path: float = 0.0,
                 autogressive: bool = True):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, width)

        self.context_length = context_length
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, width))

        self.width = width
        self.layers = layers
        self.autogressive = autogressive
        attn_mask = self.build_attention_mask() if autogressive else None
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)
               ]  # stochastic depth decay rule
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask, dpr[i])
            for i in range(layers)
        ])

        self.ln_final = LayerNorm(width)

    @property
    def dim_out(self):
        return self.width

    def build_attention_mask(self):
        # lazily create causal attention mask,
        # with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, input_ids, attention_mask=None):
        key_padding_mask = (attention_mask == 0) if (
            not self.autogressive and attention_mask is not None) else None
        x = self.token_embedding(input_ids)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        for block in self.resblocks:
            x = block(x, key_padding_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x)

        return {'last_hidden_state': x}


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the
        square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 drop_path: float = 0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def attention(self,
                  x: torch.Tensor,
                  key_padding_mask: torch.Tensor = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None

        return self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        x = x + self.drop_path(
            self.attention(self.ln_1(x), key_padding_mask=key_padding_mask))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x
