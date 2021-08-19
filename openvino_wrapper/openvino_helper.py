import importlib
import inspect
from functools import partial

import onnx


def update_default_args_value(func, **updated_args):
    new_func = partial(func, **updated_args)
    return new_func


class OpenvinoExportHelper():
    """This class contains useful methods that are needed to create OpenVINO
    compatible models."""

    @staticmethod
    def __get_funtions_with_pattern(module_name, pattern):
        """Returns a list of function and function names from the specified
        module, that match a pattern."""
        module = importlib.import_module(
            '.' + module_name, package='openvino_wrapper')
        function_members = inspect.getmembers(module, inspect.isfunction)
        selected_functions = []
        for name, func in function_members:
            if pattern in name:
                selected_functions.append((name, func))
        return selected_functions

    @staticmethod
    def process_extra_symbolics_for_openvino(opset=11):
        """Registers additional symbolic functions for OpenVINO (defined in
        'symbolic.py') and applies replacement to the original symbolic
        functions.

        To use a patch, its name must start with 'get_patch_'.
        """
        assert opset >= 10
        module_name = 'symbolic'
        pattern = 'get_patch_'
        patch_functions = OpenvinoExportHelper.__get_funtions_with_pattern(
            module_name, pattern)

        domain = 'mmdet_custom'
        from torch.onnx.symbolic_registry import register_op
        print('\t Symbolic function patches:')
        for name, function in patch_functions:
            patch = function()
            opname = patch.get_operation_name()
            symbolic_function = patch.get_symbolic_func()
            register_op(opname, symbolic_function, domain, opset)
            patch.apply_patch()
            text = name.replace(pattern, '')
            print(f'Patch {text} applied')
        print()

    @staticmethod
    def rename_input_onnx(onnx_model_path, old_name, new_name):
        """Changes the input name of the model.

        Useful for use in tests from OTEDetection.
        """
        onnx_model = onnx.load(onnx_model_path)
        for node in onnx_model.graph.node:
            for i in range(len(node.input)):
                if node.input[i] == old_name:
                    node.input[i] = new_name

        for input in onnx_model.graph.input:
            if input.name == old_name:
                input.name = new_name

        onnx.save(onnx_model, onnx_model_path)

    @staticmethod
    def __apply_fixes_from_module(module_name, skip_fixes):
        """Applies all fixes from the specified file. If you want to disable a
        specific fix, then add it to 'skip_fixes'.

        To use a fix, its name must start with 'fix_'.
        """
        pattern = 'fix_'
        fix_functions = OpenvinoExportHelper.__get_funtions_with_pattern(
            module_name, pattern)

        print(f'\t Fixes {module_name}')
        for name, function in fix_functions:
            fix_name = name.replace(pattern, '')
            if name in skip_fixes:
                print(f'Fix {fix_name} skipped.')
            else:
                function()
                print(f'Fix {fix_name} applied.')
        print()

    @staticmethod
    def apply_fixes(skip_fixes=[]):
        """This function applies fixes, contained in the 'openvino_wrapper'
        package.

        If you want to disable a specific fix, then add it to 'skip_fixes'.
        """

        modules = ['mmdetection']
        for module in modules:
            OpenvinoExportHelper.__apply_fixes_from_module(module, skip_fixes)
