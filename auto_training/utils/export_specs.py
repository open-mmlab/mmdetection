import argparse
import copy
import glob
import json
import os
import os.path as osp
import pdb
import shutil
import time
from typing import Dict

from mmcv import Config, DictAction




def get_map_from_file(file_path):
    with open(file_path) as f:
        for line in reversed(f.readlines()):
            a = json.loads(line)
            if "bbox_mAP" in a and a["mode"]=="val":
                return a["bbox_mAP"]

def write_string_as_file(path: str, string: str) -> None:
    f = open(path, "w")
    f.write(string)
    f.close()

def recreate_dir(dir_name: str):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def write_detector_yaml(cfg: Config, write_dir: str, name: str) -> None:
    detector_name = "KeypointTrtDetector"
    export_name = f"{name}.onnx"
    onnx_file_path = f"/data/ml_models/models/pvt_trt/{cfg.project_name}/{export_name}"
    yaml_contents =  f"- !<{detector_name}>\n" \
                     f"  name: {name}\n" \
                     f"  input_shape: !!python/tuple {cfg.input_res}\n" \
                     f"  onnx_file_path: {onnx_file_path}\n" \
                     f"  classes: {cfg.used_classes}\n" \
                     f"  report classes: {cfg.used_classes}\n" \
                     f"  engine_cache_path: /opt/{name}.engine\n"
    write_path = os.path.join(write_dir, "detectors.yaml")
    write_string_as_file(write_path, yaml_contents)

def write_info_file(cfg: Config, write_dir: str) -> None:
    result_file = os.path.join(cfg.work_dir, "auto.log.json")
    mAP = get_map_from_file(result_file)
    train_img_folder = cfg["data"]["train"]["dataset"]["img_prefix"]
    val_img_folder = cfg["data"]["val"]["img_prefix"]
    num_tain_img = len(glob.glob(f"{train_img_folder}/*"))
    num_val_img = len(glob.glob(f"{val_img_folder}/*"))
    date_trained = time.strftime("%m.%d.%Y %H:%M:%S")
    info_file = f" -- Object Detection training info (MMDet) -- \n" \
                f"mAP score: {mAP}\n" \
                f"Num train images: {num_tain_img}\n" \
                f"Num val images: {num_val_img}\n" \
                f"Date trained: {date_trained}\n" \
                f"Jira task: {cfg.jira_task}\n" \
                f"Author: {cfg.author}\n"
    write_path = os.path.join(write_dir, "model_info.txt")
    write_string_as_file(write_path, info_file)

def copy_training_specs(cfg, write_dir):
    shutil.copy(cfg.filename, os.path.join(write_dir, "config.txt"))



def export_for_lv(args):
    cfg = Config.fromfile(args.config)

    cfg.work_dir = args.work_dir
    if args.project_name:
        cfg.project_name = args.project_name
    else:
        cfg.project_name = cfg["data"]["train"]["dataset"]["ann_file"].split("/")[1]
    cfg.jira_task = args.jira_task
    cfg.author = args.author
    export_folder = os.path.join(cfg.work_dir, "export")
    recreate_dir(export_folder)
    model_name = f"keypoint_detector_{cfg.project_name}_{time.strftime('%y%m%d')}"
    write_detector_yaml(cfg=cfg, write_dir=export_folder, name=model_name)
    write_info_file(cfg=cfg, write_dir=export_folder)
    copy_training_specs(cfg=cfg, write_dir=export_folder)
    print(f"Training info exported successfully to: {export_folder}")
    model_output_path = os.path.join(export_folder, f"{model_name}.onnx")
    return model_output_path

# python3 auto_training/utils/export_specs.py work_dirs/fstb0/auto.py --work-dir work_dirs/fstb0 --project_name FSTTEST --author chris --jira_task OR-99888
