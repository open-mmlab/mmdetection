import torch
import torch.nn as nn

# from mmpl.registry import MODELS
from mmdet.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class Sirens(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels=3,
                 base_channels=256,
                 num_inner_layers=2,
                 is_residual=True
                 ):
        super(Sirens, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_inner_layers = num_inner_layers
        self.is_residual = is_residual

        self.first_coord = nn.Linear(in_channels, base_channels)
        self.inner_coords = nn.ModuleList(nn.Linear(base_channels, base_channels) for _ in range(self.num_inner_layers))
        self.last_coord = nn.Linear(base_channels, out_channels)

    def forward(self, x):
        x = self.first_coord(x)
        x = torch.sin(x)
        for idx in range(self.num_inner_layers):
            residual = x
            x = self.inner_coords[idx](x)
            if self.is_residual:
                x = x + residual
            x = torch.sin(x)
        x = self.last_coord(x)
        return x


@MODELS.register_module()
class ModulatedSirens(BaseModule):
    def __init__(self,
                 num_inner_layers,
                 in_dim,
                 modulation_dim,
                 out_dim=3,
                 base_channels=256,
                 is_residual=True
                 ):
        super(ModulatedSirens, self).__init__()
        self.in_dim = in_dim
        self.num_inner_layers = num_inner_layers
        self.is_residual = is_residual

        self.first_mod = nn.Sequential(
            nn.Conv2d(modulation_dim, base_channels, 1),
            nn.ReLU()
        )
        self.first_coord = nn.Conv2d(in_dim, base_channels, 1)

        self.inner_mods = nn.ModuleList()
        self.inner_coords = nn.ModuleList()
        for _ in range(self.num_inner_layers):
            self.inner_mods.append(
                nn.Sequential(
                    nn.Conv2d(modulation_dim+base_channels+base_channels, base_channels, 1),
                    nn.ReLU()
                )
            )
            self.inner_coords.append(
                    nn.Conv2d(base_channels, base_channels, 1)
            )
        self.last_coord = nn.Sequential(
            # nn.Conv2d(base_channels, base_channels//2, 1),
            # nn.ReLU(),
            nn.Conv2d(base_channels, out_dim, 1)
        )

    def forward(self, x, ori_modulations=None):
        modulations = self.first_mod(ori_modulations)
        x = self.first_coord(x)  # B 2 H W -> B C H W
        x = x + modulations
        x = torch.sin(x)
        for i_layer in range(self.num_inner_layers):
            modulations = self.inner_mods[i_layer](
                torch.cat((ori_modulations, modulations, x), dim=1))
            # modulations = self.inner_mods[i_layer](
            #     torch.cat((ori_modulations, x), dim=1))
            residual = self.inner_coords[i_layer](x)
            residual = residual + modulations
            residual = torch.sin(residual)
            if self.is_residual:
                x = x + residual
            else:
                x = residual
        x = self.last_coord(x)
        return x
