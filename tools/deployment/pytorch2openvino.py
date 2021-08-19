import argparse
import os
import subprocess
from subprocess import CalledProcessError, run

import onnx
from openvino_wrapper import OpenvinoExportHelper, update_default_args_value


def parse_args_wrapper(args_list=None):
    parser = argparse.ArgumentParser(
        description='pytorch2onnx wrapper to handle additional parameters.')
    parser.add_argument(
        '--not_strip_doc_string',
        action='store_true',
        help='If is, does not strip the field "doc_string"'
        'from the exported model, which information about the stack trace.')
    parser.add_argument(
        '--skip_fixes',
        type=str,
        nargs='+',
        default=[],
        help='The names of the fixes to be skipped.')

    args, other_args_list = parser.parse_known_args(args=args_list)
    return args, other_args_list


def parse_args(args_list=None):
    wrapper_args, args_list = parse_args_wrapper(args_list)
    from pytorch2onnx import parse_args as parse_args_pytorch2onnx
    pytorch2onnx_args = parse_args_pytorch2onnx(args_list)
    return wrapper_args, pytorch2onnx_args


def run_pytorch2onnx(args):
    from pytorch2onnx import main
    main(args)


def check_output_path(model_path):
    output_dir = os.path.dirname(model_path)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_input_shape(config):
    shape = []
    input_size = config.get('input_size', None)
    test_pipeline = config.get('test_pipeline', None)
    if input_size is not None:
        shape = [1, 3, input_size, input_size]
    elif test_pipeline is not None:
        img_scale = test_pipeline[1]['img_scale']
        shape = [1, 3, img_scale[1], img_scale[0]]
    else:
        shape = [1, 3, 800, 1344]

    return shape


def get_mean_and_scale_values(config):
    img_norm_cfg = config.get('img_norm_cfg', None)
    if img_norm_cfg is not None:
        return img_norm_cfg['mean'], img_norm_cfg['std']
    else:
        raise AttributeError(f'File {config} does not contain "img_norm_cfg"')


def run_mo(config_path, model_onnx_path):
    output_dir = check_output_path(model_onnx_path)
    from mmcv import Config
    config = Config.fromfile(config_path)
    mean_values, scale_values = get_mean_and_scale_values(config)
    output = ','.join(
        set(out.name for out in onnx.load(model_onnx_path).graph.output))
    input_shape = get_input_shape(config)

    try:
        run('mo.py -h',
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True,
            check=True)
    except CalledProcessError:
        raise RuntimeError(
            'OpenVINO Model Optimizer is not found or configured improperly')

    mo_args = f'--input_model="{model_onnx_path}" '\
              f'--mean_values="{mean_values}" ' \
              f'--scale_values="{scale_values}" ' \
              f'--output_dir="{output_dir}" ' \
              f'--output="{output}" ' \
              f'--input_shape="{input_shape}" ' \
              f'--reverse_input_channels ' \
              f'--disable_fusing '
    command = f'mo.py {mo_args}'
    print(f'Args for mo.py: {command}')
    mo_output = run(command, capture_output=True, shell=True, check=True)
    print(mo_output.stdout.decode())
    print(mo_output.stderr.decode())
    model_xml = os.path.join(output_dir, 'config.xml')
    model_bin = os.path.join(output_dir, 'config.bin')
    print(f'Successfully exported OpenVINO model: {model_xml}, {model_bin}')


def update_strip_doc_string():
    from torch import onnx
    onnx.export = update_default_args_value(
        onnx.export, strip_doc_string=False)


def main(args=None):
    wrapper_args, pytorch2onnx_args = parse_args(args)

    check_output_path(pytorch2onnx_args.output_file)

    OpenvinoExportHelper.apply_fixes(wrapper_args.skip_fixes)

    if wrapper_args.not_strip_doc_string:
        update_strip_doc_string()

    OpenvinoExportHelper.process_extra_symbolics_for_openvino(
        pytorch2onnx_args.opset_version)

    run_pytorch2onnx(pytorch2onnx_args)

    onnx_model_path = pytorch2onnx_args.output_file

    run_mo(pytorch2onnx_args.config, onnx_model_path)


if __name__ == '__main__':
    main()
