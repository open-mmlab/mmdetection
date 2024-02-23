from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from .common import LayerNorm, Transformer


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool
        # is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool
            # , and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict([('-1', nn.AvgPool2d(stride)),
                             ('0',
                              nn.Conv2d(
                                  inplanes,
                                  planes * self.expansion,
                                  1,
                                  stride=1,
                                  bias=False)),
                             ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, return_tokens=False, attn_masks=None):
        N, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1],
                      x.shape[2] * x.shape[3]).permute(2, 0,
                                                       1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        if return_tokens:
            tokens = self.c_proj(self.v_proj(x[1:])).permute(
                1, 2, 0).contiguous().view(N, -1, H, W)
        else:
            tokens = None

        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
            attn_mask=attn_masks)

        return x[0], tokens


@MODELS.register_module()
class CLIPResLayer4(BaseModule):

    def __init__(self,
                 inplanes,
                 planes,
                 blocks,
                 stride=1,
                 freeze=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze = freeze
        layers = [Bottleneck(inplanes, planes, stride)]

        inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(inplanes, planes))

        self.layer4 = nn.Sequential(*layers)

    def init_weights(self):
        super().init_weights()
        if self.freeze:
            print_log('Freeze the weights of CLIPResLayer4.')
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.layer4(x)


@MODELS.register_module()
class CLIPResNet(BaseModule):
    """A ResNet class that is similar to torchvision's but contains the
    following changes:

    - There are now 3 "stem" convolutions as opposed to 1, with an average pool
      instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is
      prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self,
                 layers,
                 output_dim,
                 heads,
                 input_resolution=224,
                 width=64,
                 freeze=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.freeze = freeze
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.attn_resolution = input_resolution // 32

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(
            width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        # this is a *mutable* variable used during construction
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.num_heads = heads
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim,
                                        heads, output_dim)

    def init_weights(self):
        super().init_weights()
        if self.freeze:
            print_log('Freeze the weights of CLIPResNet.')
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def encode_image(self,
                     x: torch.Tensor,
                     normalize=True,
                     return_tokens=False,
                     attn_masks=None):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                             (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x, image_tokens = self.attnpool(
            x, return_tokens=return_tokens, attn_masks=attn_masks)
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
        if return_tokens:
            assert image_tokens is not None
            return x, image_tokens
        else:
            return x


@MODELS.register_module()
class CLIPViT(BaseModule):

    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 freeze=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.freeze = freeze
        self.input_resolution = input_resolution
        self.attn_resolution = input_resolution // patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.num_heads = heads

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def init_weights(self):
        super().init_weights()
        if self.freeze:
            print_log('Freeze the weights of CLIPViT.')
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def rescale_positional_embedding(self, out_size, dtype):
        rescaled_positional_embedding = self.positional_embedding.new_zeros(
            1 + out_size**2, self.positional_embedding.shape[1])
        rescaled_positional_embedding[0] = self.positional_embedding[0]
        pe_2d = self.positional_embedding[1:].T.contiguous().view(
            1, -1, self.pe_grid_size, self.pe_grid_size)
        pe_2d = F.interpolate(
            pe_2d, (out_size, out_size),
            mode='bilinear').view(-1, out_size**2)
        rescaled_positional_embedding[1:] = pe_2d.T.contiguous()

        return rescaled_positional_embedding.to(dtype=dtype)

    def encode_image(self,
                     x: torch.Tensor,
                     normalize=True,
                     return_tokens=False,
                     attn_masks=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid_size = x.shape[-1]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        if grid_size == self.attn_resolution:
            pe = self.positional_embedding.to(x.dtype)
        else:
            pe = self.rescale_positional_embedding(
                out_size=grid_size, dtype=x.dtype)
        x = x + pe
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, image_tokens = self.transformer(
            x,
            return_tokens=return_tokens,
            cls_indices=0,
            attn_masks=attn_masks)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        if normalize:
            x = F.normalize(x, p=2, dim=-1)

        if return_tokens:
            image_tokens = image_tokens.permute(1, 0, 2)
            image_tokens = self.ln_post(image_tokens)
            if self.proj is not None:
                image_tokens = image_tokens @ self.proj

            # return the processed image token embeddings
            image_tokens = image_tokens[:, 1:].permute(0, 2, 1).contiguous()
            image_tokens = image_tokens.view(x.shape[0], -1, grid_size,
                                             grid_size)
            return x, image_tokens
        else:
            assert image_tokens is None
            return x
