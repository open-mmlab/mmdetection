import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init


class SSDNeck(nn.Module):

    arch_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256),
    }

    def __init__(self,
                 input_size,
                 l2_dim=512,
                 in_dim=1024,
                 l2_norm_scale=20.):
        super(SSDNeck, self).__init__()
        self.in_dim = in_dim
        self.neck = self._make_layer(self.arch_setting[input_size])
        self.l2norm = L2Norm(l2_dim, l2_norm_scale)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def _make_layer(self, out_dims):
        layers = []
        kernel_sizes = (1, 3)
        num_layers = 0
        out_dim = None
        for i in range(len(out_dims)):
            if self.in_dim == 'S':
                self.in_dim = out_dim
                continue
            k = kernel_sizes[num_layers % 2]
            if out_dims[i] == 'S':
                out_dim = out_dims[i + 1]
                conv = nn.Conv2d(self.in_dim, out_dim, k, stride=2, padding=1)
            else:
                out_dim = out_dims[i]
                conv = nn.Conv2d(self.in_dim, out_dim, k, stride=1, padding=0)
            layers.append(conv)
            self.in_dim = out_dims[i]
            num_layers += 1

        return nn.Sequential(*layers)

    def forward(self, inputs):
        outs = []
        conv4_feats, conv5_feats = inputs
        outs.append(self.l2norm(conv4_feats))
        outs.append(conv5_feats)

        x = conv5_feats
        for i, v in enumerate(self.neck):
            x = F.relu(v(x), inplace=True)
            if i % 2 == 1:
                outs.append(x)
        return outs


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        constant_init(self, scale)

    def forward(self, x):
        norm = x.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return self.weight[None, :, None, None].expand_as(x) * x / norm
