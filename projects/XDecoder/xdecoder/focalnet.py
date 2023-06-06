import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.cnn.bricks import DropPath

from mmdet.registry import MODELS

# modified from https://github.com/microsoft/X-Decoder/blob/main/xdecoder/backbone/focal_dw.py # noqa


@MODELS.register_module()
class FocalNet(nn.Module):

    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.3,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        out_indices=[0, 1, 2, 3],
        frozen_stages=-1,
        focal_levels=[3, 3, 3, 3],
        focal_windows=[3, 3, 3, 3],
        use_pre_norms=[False, False, False, False],
        use_conv_embed=True,
        use_postln=True,
        use_postln_in_modulation=False,
        scaling_modulator=True,
        use_layerscale=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            use_conv_embed=use_conv_embed,
            is_stem=True,
            use_pre_norm=False)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchEmbed if
                (i_layer < self.num_layers - 1) else None,
                focal_window=focal_windows[i_layer],
                focal_level=focal_levels[i_layer],
                use_pre_norm=use_pre_norms[i_layer],
                use_conv_embed=use_conv_embed,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                scaling_modulator=scaling_modulator,
                use_layerscale=use_layerscale,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in self.out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x):
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)

        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W,
                                 self.num_features[i]).permute(0, 3, 1,
                                                               2).contiguous()
                outs['res{}'.format(i + 2)] = out
        return outs


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FocalModulation(nn.Module):
    """Focal Modulation.

    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
    """

    def __init__(self,
                 dim,
                 proj_drop=0.,
                 focal_level=2,
                 focal_window=7,
                 focal_factor=2,
                 use_postln_in_modulation=False,
                 scaling_modulator=False):

        super().__init__()
        self.dim = dim

        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.scaling_modulator = scaling_modulator

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=True)
        self.h = nn.Conv2d(
            dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim,
                        dim,
                        kernel_size=kernel_size,
                        stride=1,
                        groups=dim,
                        padding=kernel_size // 2,
                        bias=False),
                    nn.GELU(),
                ))

    def forward(self, x):
        """Forward function.

        Args:
            x: input features with shape of (B, H, W, C)
        """
        B, nH, nW, C = x.shape
        x = self.f(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        ctx_all = 0
        for level in range(self.focal_level):
            ctx = self.focal_layers[level](ctx)
            ctx_all = ctx_all + ctx * gates[:, level:level + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        if self.scaling_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


class FocalModulationBlock(nn.Module):
    """Focal Modulation Block.

    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.
            Default: nn.LayerNorm
        focal_level (int): number of focal levels
        focal_window (int): focal kernel size at level 1
    """

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 focal_level=2,
                 focal_window=9,
                 use_postln=False,
                 use_postln_in_modulation=False,
                 scaling_modulator=False,
                 use_layerscale=False,
                 layerscale_value=1e-4):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_postln = use_postln
        self.use_layerscale = use_layerscale

        self.dw1 = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.modulation = FocalModulation(
            dim,
            focal_window=self.focal_window,
            focal_level=self.focal_level,
            proj_drop=drop,
            use_postln_in_modulation=use_postln_in_modulation,
            scaling_modulator=scaling_modulator)

        self.dw2 = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

        self.H = None
        self.W = None

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(
                layerscale_value * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                layerscale_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = x + self.dw1(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        shortcut = x
        if not self.use_postln:
            x = self.norm1(x)
        x = x.view(B, H, W, C)

        # FM
        x = self.modulation(x).view(B, H * W, C)
        x = shortcut + self.drop_path(self.gamma_1 * x)
        if self.use_postln:
            x = self.norm1(x)

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = x + self.dw2(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        if not self.use_postln:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_2 * self.mlp(x))
            x = self.norm2(x)

        return x


class BasicLayer(nn.Module):
    """A basic focal modulation layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate.
            Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer.
            Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the
            end of the layer. Default: None
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        use_conv_embed (bool): Use overlapped convolution for patch
            embedding or now. Default: False
        use_checkpoint (bool): Whether to use checkpointing to save memory.
            Default: False
    """

    def __init__(
        self,
        dim,
        depth,
        mlp_ratio=4.,
        drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        focal_window=9,
        focal_level=2,
        use_conv_embed=False,
        use_postln=False,
        use_postln_in_modulation=False,
        scaling_modulator=False,
        use_layerscale=False,
        use_checkpoint=False,
        use_pre_norm=False,
    ):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            FocalModulationBlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                focal_window=focal_window,
                focal_level=focal_level,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                scaling_modulator=scaling_modulator,
                use_layerscale=use_layerscale,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                patch_size=2,
                in_chans=dim,
                embed_dim=2 * dim,
                use_conv_embed=use_conv_embed,
                norm_layer=norm_layer,
                is_stem=False,
                use_pre_norm=use_pre_norm)

        else:
            self.downsample = None

    def forward(self, x, H, W):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x_reshaped = x.transpose(1, 2).view(x.shape[0], x.shape[-1], H, W)
            x_down = self.downsample(x_reshaped)
            x_down = x_down.flatten(2).transpose(1, 2)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels.
            Default: 96.
        norm_layer (nn.Module, optional): Normalization layer.
            Default: None
        use_conv_embed (bool): Whether use overlapped convolution for
            patch embedding. Default: False
        is_stem (bool): Is the stem block or not.
    """

    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None,
                 use_conv_embed=False,
                 is_stem=False,
                 use_pre_norm=False):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_pre_norm = use_pre_norm

        if use_conv_embed:
            # if we choose to use conv embedding,
            # then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7
                padding = 3
                stride = 4
            else:
                kernel_size = 3
                padding = 1
                stride = 2
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)
        else:
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if self.use_pre_norm:
            if norm_layer is not None:
                self.norm = norm_layer(in_chans)
            else:
                self.norm = None
        else:
            if norm_layer is not None:
                self.norm = norm_layer(embed_dim)
            else:
                self.norm = None

    def forward(self, x):
        """Forward function."""
        B, C, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x,
                      (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        if self.use_pre_norm:
            if self.norm is not None:
                x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
                x = self.norm(x).transpose(1, 2).view(B, C, H, W)
            x = self.proj(x)
        else:
            x = self.proj(x)  # B C Wh Ww
            if self.norm is not None:
                Wh, Ww = x.size(2), x.size(3)
                x = x.flatten(2).transpose(1, 2)
                x = self.norm(x)
                x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x
