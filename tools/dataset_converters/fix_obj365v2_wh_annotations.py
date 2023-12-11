import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert Objects365 annotations into MS Coco format"
    )
    parser.add_argument("--root-dir", default="data/objects365v2/", help="path to objects365 data", type=str)
    parser.add_argument(
        "--apply_exif",
        dest="apply_exif",
        action="store_true",
        help="apply the exif orientation correctly",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["val", "train"],
        choices=["train", "val", "test", "minival"],
        help="subsets to convert",
    )
    parser.add_argument("--image_info_path", default='data/objects365v2/zhiyuan_objv2_train_info.txt',type=str, help="image_info_path")
    args = parser.parse_args()
    return args


args = parse_args()
root_dir = args.root_dir

if args.apply_exif:
    print("-" * 60)
    print("We will apply exif orientation...")
    print("-" * 60)

if not isinstance(args.subsets, list):
    args.subsets = [args.subsets]


for subset in args.subsets:
    # Convert annotations
    print("converting {} data".format(subset))

    # Select correct source files for each subset
    if subset == "train":
        json_name = "zhiyuan_objv2_train.json"
    elif subset == "val":
        json_name = "zhiyuan_objv2_val.json"
    elif subset == "minival":
        json_name = "zhiyuan_objv2_val.json"

    # Load original annotations
    print("loading original annotations ...")
    json_path = os.path.join(root_dir, "annotations", json_name)
    json_data = json.load(open(json_path, "r"))
    print("loading original annotations ... Done")

    print(json_data.keys())
    oi = {}

    # Add basic dataset info
    print("adding basic dataset info")

    # Add license information
    print("adding basic license info")
    oi["licenses"] = json_data["licenses"]

    # Convert category information
    print("converting category info")
    oi["categories"] = json_data["categories"]

    # Convert image mnetadata
    print("converting image info ...")
    images = json_data["images"]
    if subset == "minival":
        images = images[:5000]
    print(f"{len(images)} images get")
    rm_image_ids = []

    if args.apply_exif:
        image_info = {}
        with open(args.image_info_path, "r") as f:
            for line in f.readlines():
                line = line.strip().split()
                image_id, file_name, height, width, channel = line

                image_id = int(image_id)
                height = int(height)
                width = int(width)

                image_info[image_id] = [file_name, height, width]

        print(f"{len(image_info)} image_info get")

        new_images = []
        for img in tqdm(images):
            image_id = img["id"]

            if image_id not in image_info.keys():
                rm_image_ids.append(image_id)
                print("removing", img)
                continue

            file_name, height, width = image_info[image_id]

            assert file_name == img["file_name"]

            if width != img["width"] or height != img["height"]:
                print("before exif correction: ", img)
                img["width"], img["height"] = width, height
                print("after exif correction: ", img)

            new_images.append(img)
        images = new_images

    oi["images"] = images
    print(f"{len(images)} images keep")

    # Convert instance annotations
    print("converting annotations ...")
    annotations = json_data["annotations"]
    print(f"{len(annotations)} annotations get")

    annotations = [ann for ann in annotations if ann["image_id"] not in rm_image_ids]
    if subset == "minival":
        keep_image_ids = [img["id"] for img in images]
        annotations = [ann for ann in annotations if ann["image_id"] in keep_image_ids]

    oi["annotations"] = annotations
    print(f"{len(annotations)} annotations keep")

    # Write annotations into .json file
    json_path = os.path.join(root_dir, "annotations/", "zhiyuan_objv2_train_fixwh.json")
    print("writing output to {}".format(json_path))
    json.dump(oi, open(json_path, "w"))
    print("Done")
