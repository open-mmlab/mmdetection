from geffnet import config
from geffnet.activations.activations_autofn import *
from geffnet.activations.activations_jit import *
from geffnet.activations.activations import *


_ACT_FN_DEFAULT = dict(
    swish=swish,
    mish=mish,
    relu=F.relu,
    relu6=F.relu6,
    sigmoid=sigmoid,
    tanh=tanh,
    hard_sigmoid=hard_sigmoid,
    hard_swish=hard_swish,
)

_ACT_FN_AUTO = dict(
    swish=swish_auto,
    mish=mish_auto,
)

_ACT_FN_JIT = dict(
    swish=swish_jit,
    mish=mish_jit,
    #hard_swish=hard_swish_jit,
    #hard_sigmoid_jit=hard_sigmoid_jit,
)

_ACT_LAYER_DEFAULT = dict(
    swish=Swish,
    mish=Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    sigmoid=Sigmoid,
    tanh=Tanh,
    hard_sigmoid=HardSigmoid,
    hard_swish=HardSwish,
)

_ACT_LAYER_AUTO = dict(
    swish=SwishAuto,
    mish=MishAuto,
)

_ACT_LAYER_JIT = dict(
    swish=SwishJit,
    mish=MishJit,
    #hard_swish=HardSwishJit,
    #hard_sigmoid=HardSigmoidJit
)

_OVERRIDE_FN = dict()
_OVERRIDE_LAYER = dict()


def add_override_act_fn(name, fn):
    global _OVERRIDE_FN
    _OVERRIDE_FN[name] = fn


def update_override_act_fn(overrides):
    assert isinstance(overrides, dict)
    global _OVERRIDE_FN
    _OVERRIDE_FN.update(overrides)


def clear_override_act_fn():
    global _OVERRIDE_FN
    _OVERRIDE_FN = dict()


def add_override_act_layer(name, fn):
    _OVERRIDE_LAYER[name] = fn


def update_override_act_layer(overrides):
    assert isinstance(overrides, dict)
    global _OVERRIDE_LAYER
    _OVERRIDE_LAYER.update(overrides)


def clear_override_act_layer():
    global _OVERRIDE_LAYER
    _OVERRIDE_LAYER = dict()


def get_act_fn(name='relu'):
    """ Activation Function Factory
    Fetching activation fns by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if name in _OVERRIDE_FN:
        return _OVERRIDE_FN[name]
    if not config.is_exportable() and not config.is_scriptable():
        # If not exporting or scripting the model, first look for a JIT optimized version
        # of our activation, then a custom autograd.Function variant before defaulting to
        # a Python or Torch builtin impl
        if name in _ACT_FN_JIT:
            return _ACT_FN_JIT[name]
        if name in _ACT_FN_AUTO:
            return _ACT_FN_AUTO[name]
    return _ACT_FN_DEFAULT[name]


def get_act_layer(name='relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if name in _OVERRIDE_LAYER:
        return _OVERRIDE_LAYER[name]
    if not config.is_exportable() and not config.is_scriptable():
        if name in _ACT_LAYER_JIT:
            return _ACT_LAYER_JIT[name]
        if name in _ACT_LAYER_AUTO:
            return _ACT_LAYER_AUTO[name]
    return _ACT_LAYER_DEFAULT[name]


