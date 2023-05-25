import torch
from torch import nn
import os
from transformers import CLIPTokenizer
from .prompt_engineering import prompt_engineering, get_prompt_templates


class LanguageEncoder(nn.Module):

    def __init__(
            self,
            tokenizer='openai/clip-vit-base-patch32',
            dim_lang=512,
            dim_projection=512,
            max_token_num=77,
    ):
        super().__init__()
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_special_tokens({'cls_token':  self.tokenizer.eos_token})

        self.lang_encoder = Transformer()

        self.lang_proj = nn.Parameter(torch.empty(dim_lang, dim_projection))
        # trunc_normal_(self.lang_proj, std=.02)

        self.max_token_num = max_token_num
        self.logit_scale = nn.Parameter(torch.ones([]))

        # # captioning & retrieval
        # for key, value in queue_operator.items():
        #     self.register_buffer(key, value)

    def get_text_embeddings(self, class_names, name='default', is_eval=False, add_bgd=False, prompt=True, norm=True):
        if not is_eval:
            if prompt:
                # randomly sample one template 随机选择一个？ 那不会每次结果不一样吗？
                arbitary_concepts = [
                    prompt_engineering(
                        class_names[label].replace('-other', '').replace('-merged', '').replace('-stuff', ''),
                        topk=10000, suffix='.') \
                    for label in range(len(class_names))
                ]
                if add_bgd:
                    arbitary_concepts.append("A background in coco.")
            else:
                arbitary_concepts = class_names

            input_ids = []
            attention_masks = []
            for txt in arbitary_concepts:
                tokens = self.tokenizer(
                    txt, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                )
                tokens['input_ids'].squeeze_()
                tokens['attention_mask'].squeeze_()

                input_ids.append(tokens['input_ids'])
                attention_masks.append(tokens['attention_mask'])

            arbitary_tokens = torch.stack(input_ids)
            arbitary_attention_masks = torch.stack(attention_masks)

            text_emb = self.forward_language((arbitary_tokens.cuda(), arbitary_attention_masks.cuda()), norm=norm)
            setattr(self, '{}_text_embeddings'.format(name), text_emb)
        else:
            with torch.no_grad():
                def extract_mean_emb(txts):
                    tokens = self.tokenizer(  # CLIP tokenizer
                        txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                    )  # 每个文本会生成 81 条 prompt
                    clss_embedding = self.forward_language(
                        (tokens['input_ids'].cuda(), tokens['attention_mask'].cuda()), norm=norm)  # 计算文本嵌入
                    clss_embedding = clss_embedding.mean(dim=0)  # 求平均，变成 512 维
                    clss_embedding /= clss_embedding.norm()
                    return clss_embedding

                # 用了 clip 的 集成模板
                templates = get_prompt_templates()
                clss_embeddings = []
                if prompt:
                    for clss in class_names:
                        txts = [template.format(clss.replace('-other', '').replace('-merged', '').replace('-stuff', ''))
                                for template in templates]
                        clss_embeddings.append(extract_mean_emb(txts))
                else:
                    clss_embeddings.append(extract_mean_emb(class_names))

                if add_bgd:  # false
                    txts = ["A background in coco."]
                    clss_embeddings.append(extract_mean_emb(txts))

                text_emb = torch.stack(clss_embeddings, dim=0)  # 10, 512, 10 表示 输入的类别数，包括背景
                setattr(self, '{}_text_embeddings'.format(name), text_emb)

    def get_text_token_embeddings(self, txts, name='default', token=False, norm=False):
        if not token:
            tokens = self.tokenizer(
                txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
            )
            tokens = {key: value.cuda() for key, value in tokens.items()}
        else:
            tokens = txts
        token_emb, class_emb = self.forward_language_token((tokens['input_ids'], tokens['attention_mask']), norm=norm)
        ret = {"tokens": tokens,
               "token_emb": token_emb,
               "class_emb": class_emb, }
        setattr(self, '{}_token_embeddings'.format(name), ret)
        return ret

    def forward_language(self, texts, norm=True):
        x = self.lang_encoder(*texts)
        x = x['last_hidden_state']  # 81,77,512

        x = x[torch.arange(x.size(0)), texts[0].argmax(dim=-1)]  # 取最后的 token 对应的输出，81,512

        x = x @ self.lang_proj
        if norm:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-7)
        return x

    def forward_language_token(self, texts, norm=False):
        x = self.lang_encoder(*texts)
        token_x = x['last_hidden_state']

        class_x = token_x[torch.arange(token_x.size(0)), texts[0].argmax(dim=-1)]

        class_x = class_x @ self.lang_proj
        token_x = token_x @ self.lang_proj

        if norm:
            class_x = class_x / (class_x.norm(dim=-1, keepdim=True) + 1e-7)
            token_x = token_x / (token_x.norm(dim=-1, keepdim=True) + 1e-7)

        return token_x, class_x

    def compute_similarity(self, v_emb, name='default', fake=False):
        if fake:
            return None
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
        t_emb = getattr(self, '{}_text_embeddings'.format(name))
        output = self.logit_scale.exp() * v_emb @ t_emb.unsqueeze(0).transpose(1, 2)
        return output


class Transformer(nn.Module):
    def __init__(self,
                 context_length: int=77,
                 vocab_size: int=49408,
                 width: int=512,
                 layers: int=12,
                 heads: int=8,
                 drop_path: float = 0.0,
                 autogressive: bool =True):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, width)

        self.context_length = context_length
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, width)
        )

        self.width = width
        self.layers = layers
        self.autogressive = autogressive
        attn_mask = self.build_attention_mask() if autogressive else None
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]  # stochastic depth decay rule
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, attn_mask, dpr[i])
                for i in range(layers)
            ]
        )

        self.ln_final = LayerNorm(width)

        # trunc_normal_(self.positional_embedding, std=.02)
        # trunc_normal_(self.token_embedding.weight, std=.02)
        # self.apply(self._init_weights)

    @property
    def dim_out(self):
        return self.width

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'positional_embedding',
            'token_embedding',
        }

    def forward(self, input_ids, attention_mask=None):
        key_padding_mask = (attention_mask == 0) if (not self.autogressive and attention_mask is not None) else None
        # key_padding_mask = (input_ids == 0) if not self.autogressive else None
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
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
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

from collections import OrderedDict
from mmcv.cnn.bricks import DropPath


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
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None


        return self.attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=self.attn_mask
        )[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        x = x + self.drop_path(self.attention(self.ln_1(x), key_padding_mask=key_padding_mask))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x
