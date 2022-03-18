from mmcls.models import VisionTransformer as _VisionTransformer
from torch import nn
import torch
from ..builder import BACKBONES


@BACKBONES
class VisionTransformer(_VisionTransformer):

    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 num_groups=8,
                 out_indices=(2, 5, 8, 11),
                 drop_rate=0,
                 drop_path_rate=0,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 output_cls_token=False,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            output_cls_token=output_cls_token,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        self.res_modify_block_0 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 3, 2),
            nn.GroupNorm(num_groups, self.embed_dims), nn.GELU(),
            nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 3, 2))

        self.res_modify_block_1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 3, 2))

        self.res_modify_block_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, 3, 2))

    def forward(self, x):

        B = x.shape[0]
        x = self.patch_embed(x)
        patch_resolution = self.patch_embed.patches_resolution

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
                patch_token = patch_token.permute(0, 3, 1, 2)
                outs.append(patch_token)

        # Convert outputs of ViT to multi-scale
        for i, out in enumerate(outs):
            if i == 0:
                outs[i] = self.res_modify_block_0(out)
            elif i == 1:
                outs[i] = self.res_modify_block_1(out)
            elif i == 3:
                outs[i] = self.res_modify_block_2(out)

        return tuple(outs)