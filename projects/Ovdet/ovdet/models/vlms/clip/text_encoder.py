import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from ....utils.misc import multi_apply
from .common import LayerNorm, Transformer
from .simple_tokenizer import SimpleTokenizer


@MODELS.register_module()
class CLIPTextEncoder(BaseModule):

    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 freeze=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.freeze = freeze
        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask())
        self.tokenizer = SimpleTokenizer()
        self.sot_token = self.tokenizer.encoder['<|startoftext|>']
        self.eot_token = self.tokenizer.encoder['<|endoftext|>']
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim))

    def init_weights(self):
        super().init_weights()
        if self.freeze:
            print_log('Freeze the weights of CLIP text encoder.')
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def build_attention_mask(self, context_length=None):
        # lazily create causal attention mask,
        # with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        if context_length is None:
            context_length = self.context_length
        mask = torch.empty(context_length, context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text, normalize=True, return_word_tokens=False):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        if text.shape[1] <= self.context_length:
            x = x + self.positional_embedding[:text.shape[1]]
            custom_attn_mask = None
        else:
            pe = self.positional_embedding
            new_pe = F.interpolate(
                pe.T[None],
                size=text.shape[1],
                mode='linear',
                align_corners=True)[0].T
            custom_attn_mask = self.build_attention_mask(text.shape[1]).to(
                self.device)
            x = x + new_pe
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, word_tokens = self.transformer(
            x,
            return_tokens=return_word_tokens,
            cls_indices=text.argmax(dim=-1),
            attn_masks=custom_attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number
        # in each sequence)
        out = x[torch.arange(x.shape[0]),
                text.argmax(dim=-1)] @ self.text_projection
        x = x @ self.text_projection
        if normalize:
            out = F.normalize(out, p=2, dim=-1)

        if return_word_tokens:
            word_tokens = word_tokens.permute(1, 0, 2)  # LND -> NLD
            word_tokens = self.ln_final(word_tokens)
            word_tokens = word_tokens @ self.text_projection
            if normalize:
                word_tokens = F.normalize(word_tokens, dim=-1)
                x = F.normalize(x, dim=-1)
            word_tokens = [
                seq[1:end_token_id]
                for seq, end_token_id in zip(word_tokens, text.argmax(dim=-1))
            ]
            return out, word_tokens, x
        else:
            assert word_tokens is None
            return out

    def encode_text_endk(self, text, stepk=12, normalize=True, **kwargs):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding[:text.shape[1]]
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i in range(stepk):
            x, _ = self.transformer.resblocks[i](x)

        # x, att = self.transformer(x)
        out = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number
        # in each sequence)
        out = out[torch.arange(out.shape[0]),
                  text.argmax(dim=-1)]  # @ self.text_projection

        if normalize:
            out = F.normalize(out, dim=-1, p=2)

        return out, x, text.argmax(dim=-1)

    def encode_pseudo_text_endk(self,
                                x,
                                end_token_ids,
                                text_pe=True,
                                stepk=12,
                                normalize=True):
        if text_pe:
            x = x + self.positional_embedding[:x.shape[1]]
        else:
            for i in range(x.shape[0]):
                x[i, end_token_ids[i]:] = x[
                    i, end_token_ids[i]:] + self.positional_embedding.type(
                        self.dtype)[end_token_ids[i]:]
                x[i, 0] = x[i, 0] + self.positional_embedding[0]

        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(stepk):
            x, _ = self.transformer.resblocks[i](x)

        out = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number
        # in each sequence)

        out = out[torch.arange(out.shape[0]), end_token_ids]

        if normalize:
            out = F.normalize(out, dim=-1, p=2)

        return out, x, end_token_ids

    def encode_pseudo_text(self,
                           x,
                           end_token_ids,
                           text_pe=True,
                           normalize=True,
                           return_word_tokens=False):
        if text_pe:
            x = x + self.positional_embedding[:x.shape[1]]
        else:
            for i in range(x.shape[0]):
                x[i, end_token_ids[i]:] = x[
                    i, end_token_ids[i]:] + self.positional_embedding.type(
                        self.dtype)[end_token_ids[i]:]
                x[i, 0] = x[i, 0] + self.positional_embedding[0]

        x = x.permute(1, 0, 2)  # NLD -> LND

        num_steps = len(self.transformer.resblocks)
        for i in range(num_steps - 1):
            x, _ = self.transformer.resblocks[i](x)
        x, word_tokens = self.transformer.resblocks[-1](
            x, return_tokens=return_word_tokens, cls_indices=end_token_ids)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number
        # in each sequence)
        out = x[torch.arange(x.shape[0]), end_token_ids] @ self.text_projection

        if normalize:
            out = F.normalize(out, dim=-1, p=2)
        if return_word_tokens:
            word_tokens = word_tokens.permute(1, 0, 2)  # LND -> NLD
            word_tokens = self.ln_final(word_tokens)
            word_tokens = word_tokens @ self.text_projection
            word_tokens = [
                seq[1:end_token_id]
                for seq, end_token_id in zip(word_tokens, end_token_ids)
            ]
            return out, word_tokens
        else:
            assert word_tokens is None
            return out

    def prepare_pseudo_text_tensor(self, pseudo_tokens, valid_mask):
        # valid_mask {0.0, 1.0}
        device = pseudo_tokens.device
        num_preds, num_words, word_dim = pseudo_tokens.shape
        sot_token = self.token_embedding(
            torch.tensor([self.sot_token], device=device))
        eot_token = self.token_embedding(
            torch.tensor([self.eot_token], device=device))
        sot_token = sot_token.view(1, 1, word_dim).repeat(num_preds, 1, 1)
        eot_token = eot_token.view(1, 1, word_dim).repeat(num_preds, 1, 1)
        pseudo_tokens = torch.cat([sot_token, pseudo_tokens, eot_token], dim=1)
        num_words += 2
        assert valid_mask.shape == pseudo_tokens.shape[:2]
        pseudo_tokens_flat = pseudo_tokens.view(-1, word_dim)
        valid_mask_flat = valid_mask.view(-1)

        empty_token = self.token_embedding(torch.tensor([0], device=device))
        template_flat = empty_token.view(1, word_dim).repeat(
            num_preds * num_words, 1)

        valid_mask_zero_pad = torch.cat(
            [torch.zeros_like(valid_mask[:, :1]), valid_mask], dim=-1)
        pe_indices = (valid_mask_zero_pad > 0.0).cumsum(dim=-1)[:, :-1]
        pe_indices_flat = (pe_indices +
                           (torch.arange(num_preds, device=pe_indices.device) *
                            num_words)[:, None]).view(-1)

        template_flat[pe_indices_flat[valid_mask_flat > 0.0]] \
            = pseudo_tokens_flat[valid_mask_flat > 0.0]
        pseudo_tokens = template_flat.view(num_preds, num_words, word_dim)
        end_token_ids = (valid_mask > 0.0).sum(-1).long() - 1

        return pseudo_tokens, end_token_ids

    def prepare_pseudo_text(self, pseudo_tokens, context_length):
        device = pseudo_tokens[0].device
        sot_token = self.token_embedding(
            torch.tensor([self.sot_token],
                         device=device))  # [batch_size, n_ctx, d_model]
        eot_token = self.token_embedding(
            torch.tensor([self.eot_token], device=device))
        empty_token = self.token_embedding(torch.tensor([0], device=device))
        pseudo_tokens = [
            torch.cat([sot_token, tokens, eot_token], dim=0)
            for tokens in pseudo_tokens
        ]

        def _pad_sequence(tokens):
            if tokens.shape[0] > context_length:
                x = tokens[list(range(context_length - 1)) +
                           [tokens.shape[0] - 1]]
                end_token_id = context_length - 1
            else:
                x = torch.cat([
                    tokens,
                    empty_token.repeat(context_length - tokens.shape[0], 1)
                ],
                              dim=0)
                end_token_id = tokens.shape[0] - 1
            return x, end_token_id

        x, end_token_ids = multi_apply(_pad_sequence, pseudo_tokens)
        x = torch.stack(x, dim=0)

        return x, torch.tensor(
            end_token_ids, dtype=torch.long, device=x.device)
