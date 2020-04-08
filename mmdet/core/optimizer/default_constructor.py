import torch
from mmcv.utils import build_from_cfg
from torch.nn import GroupNorm, LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from .registry import OPTIMIZER_BUILDERS, OPTIMIZERS


@OPTIMIZER_BUILDERS.register_module
class DefaultOptimizerConstructor(object):
    """Default constructor for optimizers.

    Attributes:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
        paramwise_cfg (dict, optional): Parameter-wise options. Accepted fields
            are:
            - bias_lr_mult: It will be multiplied to the learning rate for
              all bias parameters (except for those in normalization layers).
            - bias_decay_mult: It will be multiplied to the weight decay for
              all bias parameters (except for those in normalization layers and
              depthwise conv layers).
            - norm_decay_mult: will be multiplied to the weight decay
              for all weight and bias parameters of normalization layers.
            - dwconv_decay_mult: will be multiplied to the weight decay
              for all weight and bias parameters of depthwise conv layers.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> paramwise_cfg = dict(norm_decay_mult=0.)
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)
    """

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        if not isinstance(optimizer_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict',
                            'but got {}'.format(type(optimizer_cfg)))
        self.optimizer_cfg = optimizer_cfg
        self.paramwise_cfg = {} if paramwise_cfg is None else paramwise_cfg
        self.base_lr = optimizer_cfg['lr']
        self.base_wd = optimizer_cfg.get('weight_decay', None)
        self._validate_cfg()

    def _validate_cfg(self):
        if not isinstance(self.paramwise_cfg, dict):
            raise TypeError('paramwise_cfg should be None or a dict, '
                            'but got {}'.format(type(self.paramwise_cfg)))
        # get base lr and weight decay
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in self.paramwise_cfg
                or 'norm_decay_mult' in self.paramwise_cfg
                or 'dwconv_decay_mult' in self.paramwise_cfg):
            if self.base_wd is None:
                raise ValueError('base_wd should not be None')

    def add_params(self, params, module):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """
        # get param-wise options
        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', 1.)
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', 1.)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', 1.)
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', 1.)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module,
                             (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (
            isinstance(module, torch.nn.Conv2d)
            and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue
            # bias_lr_mult affects all bias parameters except for norm.bias
            if name == 'bias' and not is_norm:
                param_group['lr'] = self.base_lr * bias_lr_mult
            # apply weight decay policies
            if self.base_wd is not None:
                # norm decay
                if is_norm:
                    param_group[
                        'weight_decay'] = self.base_wd * norm_decay_mult
                # depth-wise conv
                elif is_dwconv:
                    param_group[
                        'weight_decay'] = self.base_wd * dwconv_decay_mult
                # bias lr and decay
                elif name == 'bias':
                    param_group[
                        'weight_decay'] = self.base_wd * bias_decay_mult
            params.append(param_group)

        for child_mod in module.children():
            self.add_params(params, child_mod)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = model.parameters()
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        # set param-wise lr and weight decay recursively
        params = []
        self.add_params(params, model)
        optimizer_cfg['params'] = params

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
