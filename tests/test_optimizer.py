import warnings

import pytest
import torch
import torch.nn as nn

from mmdet.core import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmdet.core.optimizer import build_optimizer, build_optimizer_constructor
from mmdet.core.optimizer.builder import TORCH_OPTIMIZERS


class SubModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=1, groups=2)
        self.gn = nn.GroupNorm(2, 2)
        self.param1 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.sub = SubModel()

    def forward(self, x):
        return x


class ExampleDuplicateModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=False))
        self.conv2 = nn.Sequential(nn.Conv2d(4, 2, kernel_size=1))
        self.bn = nn.BatchNorm2d(2)
        self.sub = SubModel()
        self.conv3 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=False))
        self.conv3[0] = self.conv1[0]

    def forward(self, x):
        return x


class PseudoDataParallel(nn.Module):

    def __init__(self):
        super().__init__()
        self.module = ExampleModel()

    def forward(self, x):
        return x


base_lr = 0.01
base_wd = 0.0001
momentum = 0.9


def check_default_optimizer(optimizer, model, prefix=''):
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    param_groups = optimizer.param_groups[0]
    param_names = [
        'param1', 'conv1.weight', 'conv2.weight', 'conv2.bias', 'bn.weight',
        'bn.bias', 'sub.param1', 'sub.conv1.weight', 'sub.conv1.bias',
        'sub.gn.weight', 'sub.gn.bias'
    ]
    param_dict = dict(model.named_parameters())
    assert len(param_groups['params']) == len(param_names)
    for i in range(len(param_groups['params'])):
        assert torch.equal(param_groups['params'][i],
                           param_dict[prefix + param_names[i]])


def check_optimizer(optimizer,
                    model,
                    prefix='',
                    bias_lr_mult=1,
                    bias_decay_mult=1,
                    norm_decay_mult=1,
                    dwconv_decay_mult=1,
                    bypass_duplicate=False):
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    model_parameters = list(model.parameters())
    assert len(param_groups) == len(model_parameters)
    for i, param in enumerate(model_parameters):
        param_group = param_groups[i]
        assert torch.equal(param_group['params'][0], param)
        assert param_group['momentum'] == momentum
    # param1
    param1 = param_groups[0]
    assert param1['lr'] == base_lr
    assert param1['weight_decay'] == base_wd
    # conv1.weight
    conv1_weight = param_groups[1]
    assert conv1_weight['lr'] == base_lr
    assert conv1_weight['weight_decay'] == base_wd
    # conv2.weight
    conv2_weight = param_groups[2]
    assert conv2_weight['lr'] == base_lr
    assert conv2_weight['weight_decay'] == base_wd
    # conv2.bias
    conv2_bias = param_groups[3]
    assert conv2_bias['lr'] == base_lr * bias_lr_mult
    assert conv2_bias['weight_decay'] == base_wd * bias_decay_mult
    # bn.weight
    bn_weight = param_groups[4]
    assert bn_weight['lr'] == base_lr
    assert bn_weight['weight_decay'] == base_wd * norm_decay_mult
    # bn.bias
    bn_bias = param_groups[5]
    assert bn_bias['lr'] == base_lr
    assert bn_bias['weight_decay'] == base_wd * norm_decay_mult
    # sub.param1
    sub_param1 = param_groups[6]
    assert sub_param1['lr'] == base_lr
    assert sub_param1['weight_decay'] == base_wd
    # sub.conv1.weight
    sub_conv1_weight = param_groups[7]
    assert sub_conv1_weight['lr'] == base_lr
    assert sub_conv1_weight['weight_decay'] == base_wd * dwconv_decay_mult
    # sub.conv1.bias
    sub_conv1_bias = param_groups[8]
    assert sub_conv1_bias['lr'] == base_lr * bias_lr_mult
    assert sub_conv1_bias['weight_decay'] == base_wd * dwconv_decay_mult
    # sub.gn.weight
    sub_gn_weight = param_groups[9]
    assert sub_gn_weight['lr'] == base_lr
    assert sub_gn_weight['weight_decay'] == base_wd * norm_decay_mult
    # sub.gn.bias
    sub_gn_bias = param_groups[10]
    assert sub_gn_bias['lr'] == base_lr
    assert sub_gn_bias['weight_decay'] == base_wd * norm_decay_mult


def test_default_optimizer_constructor():
    model = ExampleModel()

    with pytest.raises(TypeError):
        # optimizer_cfg must be a dict
        optimizer_cfg = []
        optim_constructor = DefaultOptimizerConstructor(optimizer_cfg)
        optim_constructor(model)

    with pytest.raises(TypeError):
        # paramwise_cfg must be a dict or None
        optimizer_cfg = dict(lr=0.0001)
        paramwise_cfg = ['error']
        optim_constructor = DefaultOptimizerConstructor(
            optimizer_cfg, paramwise_cfg)
        optim_constructor(model)

    with pytest.raises(ValueError):
        # bias_decay_mult/norm_decay_mult is specified but weight_decay is None
        optimizer_cfg = dict(lr=0.0001, weight_decay=None)
        paramwise_cfg = dict(bias_decay_mult=1, norm_decay_mult=1)
        optim_constructor = DefaultOptimizerConstructor(
            optimizer_cfg, paramwise_cfg)
        optim_constructor(model)

    # basic config with ExampleModel
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    optim_constructor = DefaultOptimizerConstructor(optimizer_cfg)
    optimizer = optim_constructor(model)
    check_default_optimizer(optimizer, model)

    # basic config with pseudo data parallel
    model = PseudoDataParallel()
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    paramwise_cfg = None
    optim_constructor = DefaultOptimizerConstructor(optimizer_cfg)
    optimizer = optim_constructor(model)
    check_default_optimizer(optimizer, model, prefix='module.')

    # basic config with DataParallel
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(ExampleModel())
        optimizer_cfg = dict(
            type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
        paramwise_cfg = None
        optim_constructor = DefaultOptimizerConstructor(optimizer_cfg)
        optimizer = optim_constructor(model)
        check_default_optimizer(optimizer, model, prefix='module.')

    # Empty paramwise_cfg with ExampleModel
    model = ExampleModel()
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    paramwise_cfg = dict()
    optim_constructor = DefaultOptimizerConstructor(optimizer_cfg,
                                                    paramwise_cfg)
    optimizer = optim_constructor(model)
    check_default_optimizer(optimizer, model)

    # Empty paramwise_cfg with ExampleModel and no grad
    model = ExampleModel()
    for param in model.parameters():
        param.requires_grad = False
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    paramwise_cfg = dict()
    optim_constructor = DefaultOptimizerConstructor(optimizer_cfg)
    optimizer = optim_constructor(model)
    check_default_optimizer(optimizer, model)

    # paramwise_cfg with ExampleModel
    model = ExampleModel()
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    paramwise_cfg = dict(
        bias_lr_mult=2,
        bias_decay_mult=0.5,
        norm_decay_mult=0,
        dwconv_decay_mult=0.1)
    optim_constructor = DefaultOptimizerConstructor(optimizer_cfg,
                                                    paramwise_cfg)
    optimizer = optim_constructor(model)
    check_optimizer(optimizer, model, **paramwise_cfg)

    # paramwise_cfg with ExampleModel, weight decay is None
    model = ExampleModel()
    optimizer_cfg = dict(type='Rprop', lr=base_lr)
    paramwise_cfg = dict(bias_lr_mult=2)
    optim_constructor = DefaultOptimizerConstructor(optimizer_cfg,
                                                    paramwise_cfg)
    optimizer = optim_constructor(model)

    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.Rprop)
    assert optimizer.defaults['lr'] == base_lr
    model_parameters = list(model.parameters())
    assert len(param_groups) == len(model_parameters)
    for i, param in enumerate(model_parameters):
        param_group = param_groups[i]
        assert torch.equal(param_group['params'][0], param)
    # param1
    assert param_groups[0]['lr'] == base_lr
    # conv1.weight
    assert param_groups[1]['lr'] == base_lr
    # conv2.weight
    assert param_groups[2]['lr'] == base_lr
    # conv2.bias
    assert param_groups[3]['lr'] == base_lr * paramwise_cfg['bias_lr_mult']
    # bn.weight
    assert param_groups[4]['lr'] == base_lr
    # bn.bias
    assert param_groups[5]['lr'] == base_lr
    # sub.param1
    assert param_groups[6]['lr'] == base_lr
    # sub.conv1.weight
    assert param_groups[7]['lr'] == base_lr
    # sub.conv1.bias
    assert param_groups[8]['lr'] == base_lr * paramwise_cfg['bias_lr_mult']
    # sub.gn.weight
    assert param_groups[9]['lr'] == base_lr
    # sub.gn.bias
    assert param_groups[10]['lr'] == base_lr

    # paramwise_cfg with pseudo data parallel
    model = PseudoDataParallel()
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    paramwise_cfg = dict(
        bias_lr_mult=2,
        bias_decay_mult=0.5,
        norm_decay_mult=0,
        dwconv_decay_mult=0.1)
    optim_constructor = DefaultOptimizerConstructor(optimizer_cfg,
                                                    paramwise_cfg)
    optimizer = optim_constructor(model)
    check_optimizer(optimizer, model, prefix='module.', **paramwise_cfg)

    # paramwise_cfg with DataParallel
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(ExampleModel())
        optimizer_cfg = dict(
            type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
        paramwise_cfg = dict(
            bias_lr_mult=2,
            bias_decay_mult=0.5,
            norm_decay_mult=0,
            dwconv_decay_mult=0.1)
        optim_constructor = DefaultOptimizerConstructor(
            optimizer_cfg, paramwise_cfg)
        optimizer = optim_constructor(model)
        check_optimizer(optimizer, model, prefix='module.', **paramwise_cfg)

    # paramwise_cfg with ExampleModel and no grad
    for param in model.parameters():
        param.requires_grad = False
    optim_constructor = DefaultOptimizerConstructor(optimizer_cfg,
                                                    paramwise_cfg)
    optimizer = optim_constructor(model)
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    for i, (name, param) in enumerate(model.named_parameters()):
        param_group = param_groups[i]
        assert torch.equal(param_group['params'][0], param)
        assert param_group['momentum'] == momentum
        assert param_group['lr'] == base_lr
        assert param_group['weight_decay'] == base_wd

    # paramwise_cfg with bypass_duplicate option
    model = ExampleDuplicateModel()
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    paramwise_cfg = dict(
        bias_lr_mult=2,
        bias_decay_mult=0.5,
        norm_decay_mult=0,
        dwconv_decay_mult=0.1)
    with pytest.raises(ValueError) as excinfo:
        optim_constructor = DefaultOptimizerConstructor(
            optimizer_cfg, paramwise_cfg)
        optim_constructor(model)
        assert 'some parameters appear in more than one parameter ' \
               'group' == excinfo.value

    paramwise_cfg = dict(
        bias_lr_mult=2,
        bias_decay_mult=0.5,
        norm_decay_mult=0,
        dwconv_decay_mult=0.1,
        bypass_duplicate=True)
    optim_constructor = DefaultOptimizerConstructor(optimizer_cfg,
                                                    paramwise_cfg)
    with warnings.catch_warnings(record=True) as w:
        optimizer = optim_constructor(model)
        warnings.simplefilter('always')
        assert len(w) == 1
        assert str(w[0].message) == 'conv3.0 is duplicate. It is skipped ' \
                                    'since bypass_duplicate=True'
    model_parameters = list(model.parameters())
    assert len(optimizer.param_groups) == len(model_parameters) == 11
    check_optimizer(optimizer, model, **paramwise_cfg)


def test_torch_optimizers():
    torch_optimizers = [
        'ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS',
        'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam'
    ]
    assert set(torch_optimizers).issubset(set(TORCH_OPTIMIZERS))


def test_build_optimizer_constructor():
    model = ExampleModel()
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    paramwise_cfg = dict(
        bias_lr_mult=2,
        bias_decay_mult=0.5,
        norm_decay_mult=0,
        dwconv_decay_mult=0.1)
    optim_constructor_cfg = dict(
        type='DefaultOptimizerConstructor',
        optimizer_cfg=optimizer_cfg,
        paramwise_cfg=paramwise_cfg)
    optim_constructor = build_optimizer_constructor(optim_constructor_cfg)
    optimizer = optim_constructor(model)
    check_optimizer(optimizer, model, **paramwise_cfg)

    from mmdet.core import OPTIMIZERS
    from mmcv.utils import build_from_cfg

    @OPTIMIZER_BUILDERS.register_module()
    class MyOptimizerConstructor(DefaultOptimizerConstructor):

        def __call__(self, model):
            if hasattr(model, 'module'):
                model = model.module

            conv1_lr_mult = self.paramwise_cfg.get('conv1_lr_mult', 1.)

            params = []
            for name, param in model.named_parameters():
                param_group = {'params': [param]}
                if name.startswith('conv1') and param.requires_grad:
                    param_group['lr'] = self.base_lr * conv1_lr_mult
                params.append(param_group)
            optimizer_cfg['params'] = params

            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

    paramwise_cfg = dict(conv1_lr_mult=5)
    optim_constructor_cfg = dict(
        type='MyOptimizerConstructor',
        optimizer_cfg=optimizer_cfg,
        paramwise_cfg=paramwise_cfg)
    optim_constructor = build_optimizer_constructor(optim_constructor_cfg)
    optimizer = optim_constructor(model)

    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    for i, param in enumerate(model.parameters()):
        param_group = param_groups[i]
        assert torch.equal(param_group['params'][0], param)
        assert param_group['momentum'] == momentum
    # conv1.weight
    assert param_groups[1]['lr'] == base_lr * paramwise_cfg['conv1_lr_mult']
    assert param_groups[1]['weight_decay'] == base_wd


def test_build_optimizer():
    model = ExampleModel()
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    optimizer = build_optimizer(model, optimizer_cfg)
    check_default_optimizer(optimizer, model)

    model = ExampleModel()
    optimizer_cfg = dict(
        type='SGD',
        lr=base_lr,
        weight_decay=base_wd,
        momentum=momentum,
        paramwise_cfg=dict(
            bias_lr_mult=2,
            bias_decay_mult=0.5,
            norm_decay_mult=0,
            dwconv_decay_mult=0.1))
    optimizer = build_optimizer(model, optimizer_cfg)
    check_optimizer(optimizer, model, **optimizer_cfg['paramwise_cfg'])
