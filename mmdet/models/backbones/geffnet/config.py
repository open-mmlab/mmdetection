""" Global Config and Constants
"""

__all__ = ['is_exportable', 'is_scriptable', 'set_exportable', 'set_scriptable']

# Set to True if exporting a model with Same padding via ONNX
_EXPORTABLE = False

# Set to True if wanting to use torch.jit.script on a model
_SCRIPTABLE = False


def is_exportable():
    return _EXPORTABLE


def set_exportable(value):
    global _EXPORTABLE
    _EXPORTABLE = value


def is_scriptable():
    return _SCRIPTABLE


def set_scriptable(value):
    global _SCRIPTABLE
    _SCRIPTABLE = value

