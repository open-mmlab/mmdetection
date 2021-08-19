import importlib
from functools import wraps

import torch


def fix_get_bboxes_output():
    """Because in SingleStageDetector.onnx_export, after calling
    self.bbox_head.get_bboxes, only two output values are expected: det_bboxes,
    det_labels."""

    def crop_output(function):

        @wraps(function)
        def wrapper(*args, **kwargs):
            output_list = function(*args, **kwargs)
            if len(output_list) == 2:
                return output_list
            else:
                return output_list[0]

        return wrapper

    dense_heads = importlib.import_module('mmdet.models.dense_heads')
    heads = ['FoveaHead', 'ATSSHead', 'VFNetHead', 'YOLOXHead']
    for head_name in heads:
        head_class = getattr(dense_heads, head_name)
        head_class.get_bboxes = crop_output(head_class.get_bboxes)


def fix_img_shape_type():
    """Some models (ATSS, FoveaBox) use img_metas[0]['img_shape'], which type
    is 'list'.

    To export with dynamic inputs, the type should be changed to 'tensor'.
    Can be fixed here: https://github.com/open-mmlab/mmdetection/pull/5251.
    Or you can remove support for img_metas[0]['img_shape_for_onnx'] and
    always convert values to a tensor with input dimension.
    """

    def rewrite_img_shape_in_onnx_export(function):

        @wraps(function)
        def wrapper(self, img, img_metas):
            img_metas[0]['img_shape'] = torch._shape_as_tensor(img)[2:]
            return function(self, img, img_metas)

        return wrapper

    from mmdet.models.detectors.single_stage import SingleStageDetector
    SingleStageDetector.onnx_export = rewrite_img_shape_in_onnx_export(
        SingleStageDetector.onnx_export)


# Need to fix, it does not work :(
def fix_model_device_type():
    """The VFNet model requires the DeformConv operation, which is not
    implemented for CPU tensors.

    This function changes the device type for the model and input data.
    """

    def try_to_change_device_type(generate_inputs_and_wrap_model):

        @wraps(generate_inputs_and_wrap_model)
        def wrapper(*args, **kwargs):
            model, tensor_data = generate_inputs_and_wrap_model(
                *args, **kwargs)
            if torch.cuda.is_available():
                model.cuda()
            device = next(model.parameters()).device
            tensor_data = [tensor.to(device) for tensor in tensor_data]
            return model, tensor_data

        return wrapper

    return
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    '''
    from mmdet.core.export import generate_inputs_and_wrap_model
    generate_inputs_and_wrap_model = try_to_change_device_type(
        generate_inputs_and_wrap_model)
    '''
