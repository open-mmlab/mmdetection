import os.path as osp
import sys
from importlib import import_module

config = 'python'
if isinstance(config, str):
    print("True")

config = "../../../configs/faster_rcnn_r50_fpn_1x.py"
file_name = osp.abspath(osp.expanduser(config))
print(f'file_name is {file_name}')
module_name = osp.basename(file_name)[:-3]
print(f"module_name is {module_name}")
config_dir = osp.dirname(file_name)
print(f"config_dir is {config_dir}")

sys.path.insert(0, config_dir)
mod = import_module(module_name)
for path in sys.path:
    print(path)
sys.path.pop
