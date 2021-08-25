import importlib
import inspect


class OpenvinoExportHelper():
    """This class contains useful methods that are needed to create OpenVINO
    compatible models."""

    @staticmethod
    def __get_funtions_with_pattern(module_name, pattern):
        """Returns a list of function and function names from the specified
        module, that match a pattern."""
        module = importlib.import_module(
            '.' + module_name, package='mmdet.core.export.openvino_wrapper')
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
