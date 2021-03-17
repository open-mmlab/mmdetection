from contextlib import suppress

from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmdet


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = mmdet.__version__ + '+' + get_git_hash()[:7]
    from mmcv.ops import get_compiler_version, get_compiling_cuda_version
    env_info['MMDetection Compiler'] = get_compiler_version()
    env_info['MMDetection CUDA Compiler'] = get_compiling_cuda_version()
    from mmdet.integration.nncf.utils import get_nncf_version
    env_info['NNCF'] = get_nncf_version()

    env_info['ONNX'] = None
    with suppress(ImportError):
        import onnx
        env_info['ONNX'] = onnx.__version__

    env_info['ONNXRuntime'] = None
    with suppress(ImportError):
        import onnxruntime
        env_info['ONNXRuntime'] = onnxruntime.__version__

    env_info['OpenVINO MO'] = None
    with suppress(ImportError):
        from mo.utils.version import get_version
        env_info['OpenVINO MO'] = get_version()

    env_info['OpenVINO IE'] = None
    with suppress(ImportError):
        import openvino.inference_engine as ie
        env_info['OpenVINO IE'] = ie.__version__

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
