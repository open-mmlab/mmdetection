import inspect
import logging

import mmcv
import torch

logger = logging.getLogger(__name__)


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if mmcv.is_str(obj_type):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    jit = args.pop('jit', False)

    if not jit:
        return obj_cls(**args)

    jit_dump_graph_for = args.pop('jit_dump_graph_for', False)
    jit_dump_graph = args.pop('jit_dump_graph', False)
    jit_dump_code = args.pop('jit_dump_code', False)

    orig = obj_cls(**args)
    jitted = torch.jit.script(orig)
    orig._jitted = jitted
    _restore_load_state_dict_pre_hooks(orig, jitted)
    if jit_dump_graph_for:
        logger.info(jitted.graph_for, extra=cfg)
    if jit_dump_graph:
        logger.info(jitted.graph, extra=cfg)
    if jit_dump_code:
        logger.info(jitted.code, extra=cfg)
    return jitted


def _restore_load_state_dict_pre_hooks(orig, jitted):
    for key, hook in orig._load_state_dict_pre_hooks.items():
        jitted._load_state_dict_pre_hooks[key] = hook
    orig_children = dict(orig.named_children())
    for name, child in jitted.named_children():
        orig_child = orig_children[name]
        _restore_load_state_dict_pre_hooks(orig_child, child)
