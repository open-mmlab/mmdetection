# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmengine
import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import random

# NuScenes categories for 2D detection
NUSCENES_CATEGORIES = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "truck"},
    {"id": 3, "name": "trailer"},
    {"id": 4, "name": "bus"},
    {"id": 5, "name": "construction_vehicle"},
    {"id": 6, "name": "bicycle"},
    {"id": 7, "name": "motorcycle"},
    {"id": 8, "name": "pedestrian"},
    {"id": 9, "name": "traffic_cone"},
    {"id": 10, "name": "barrier"}
]

IGNORED_CATEGORIES = [
    {"id": 5, "name": "construction_vehicle"},
    {"id": 9, "name": "traffic_cone"},
    {"id": 10, "name": "barrier"}
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert NuScenes MMDetection3D format to MMDetection format')
    parser.add_argument('--data-root', default='data/nuscenes',
                        help='NuScenes data root')
    parser.add_argument(
        '--out-dir', default='data/nuscenes/annotations', help='Output directory')
    parser.add_argument('--camera', default='CAM_FRONT',
                        help='Camera to use for 2D detection')
    parser.add_argument('--pkl-prefix', default='nuscenes_infos',
                        help='Prefix of the pkl files')
    parser.add_argument('--mini', action='store_true',
                        help='Create a mini dataset for debugging')
    parser.add_argument('--mini-size', type=int, default=50,
                        help='Number of images in mini dataset')
    parser.add_argument('--mini-prefix', default='mini_',
                        help='Prefix for mini dataset files')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print("Converting NuScenes MMDetection3D format to MMDetection format...")

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Process splits
    for split in ['train', 'val']:
        print(f"Processing {split} split...")

        # Load pickle file
        pkl_file = osp.join(args.data_root, f'{args.pkl_prefix}_{split}.pkl')
        data_infos = mmengine.load(pkl_file)

        # Initialize COCO format data
        coco_data = {
            "info": {
                "description": f"NuScenes 2D {split} set converted to COCO format",
                "version": "v1.0",
                "year": 2023,
                "contributor": "MMDetection Custom",
                "date_created": "2023"
            },
            "licenses": [{"id": 1, "name": "Attribution-NonCommercial-ShareAlike 4.0 License"}],
            "categories": NUSCENES_CATEGORIES,
            "images": [],
            "annotations": []
        }

        # Count for IDs - start from 1 not 0 to pass verification
        image_id = 1
        ann_id = 1

        # Check if data_infos has new format with metainfo and data_list
        if isinstance(data_infos, dict) and 'data_list' in data_infos:
            data_list = data_infos['data_list']
        else:
            data_list = data_infos

        # Create mini dataset if requested
        if args.mini:
            # Make sure the requested size isn't larger than the dataset
            mini_size = min(args.mini_size, len(data_list))
            # Randomly sample the dataset
            data_list = random.sample(data_list, mini_size)
            print(f"Created mini dataset with {len(data_list)} samples")

        # Process each sample
        for info in tqdm(data_list, desc=f"Processing {split} samples"):
            # Get camera image path
            if args.camera not in info['images']:
                continue

            img_info = info['images'][args.camera]
            img_path = img_info['img_path']

            # Fix path handling: Check if it's a relative path and fix it
            if not osp.isabs(img_path):
                # Try to find the correct path
                # First check if the path as-is exists
                full_path = osp.join(args.data_root, img_path)
                if not osp.exists(full_path):
                    # If not, try extracting the filename and looking in samples/CAM_XXX
                    filename = osp.basename(img_path)
                    cam_dir = args.camera
                    alternate_path = osp.join(
                        args.data_root, 'samples', cam_dir, filename)
                    if osp.exists(alternate_path):
                        full_path = alternate_path
                    else:
                        # Skip if we can't find the image
                        print(
                            f"Warning: Could not find image at {full_path} or {alternate_path}, skipping")
                        continue
                img_path = full_path

            # Verify the file exists
            if not osp.exists(img_path):
                print(f"Warning: Image {img_path} does not exist, skipping")
                continue

            try:
                # Get image dimensions
                img = Image.open(img_path)
                width, height = img.size

                # Create image entry with a relative path for COCO format
                rel_path = osp.relpath(img_path, args.data_root)
                absolute_path = osp.abspath(img_path)

                # Use a dummy URL format to satisfy validation requirements
                coco_url = f"http://example.org/nuscenes/{rel_path}"

                image_entry = {
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": rel_path,
                    "license": 1,
                    "sample_token": info.get('token', ''),
                    "coco_url": coco_url,  # Add coco_url to satisfy the validation
                    "absolute_url": absolute_path  # Add absolute path as absolute_url
                }
                coco_data["images"].append(image_entry)

                # Process 2D annotations for this image
                if 'cam_instances' in info and args.camera in info['cam_instances']:
                    cam_instances = info['cam_instances'][args.camera]

                    for instance in cam_instances:
                        # Get 2D bounding box
                        bbox = instance.get('bbox')
                        if bbox is None or len(bbox) != 4:
                            continue

                        # Get category
                        bbox_label = instance.get('bbox_label', -1)
                        # Convert label index to category ID (add 1 to match the IDs in NUSCENES_CATEGORIES)
                        category_id = bbox_label + 1

                        # dont add construction_vehicle, traffic_cone, barrier
                        if category_id in [cat['id'] for cat in IGNORED_CATEGORIES]:
                            continue

                        # assert category_id is valid
                        assert category_id in [
                            cat['id'] for cat in NUSCENES_CATEGORIES], f"Invalid category ID {category_id}"

                        # Calculate area
                        x1, y1, x2, y2 = bbox
                        area = (x2 - x1) * (y2 - y1)

                        # Skip invalid boxes
                        if x1 >= x2 or y1 >= y2 or area <= 0:
                            continue

                        # Create annotation with COCO format bbox [x, y, width, height]
                        annotation = {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "area": float(area),
                            "iscrowd": 0,
                            "segmentation": [],
                        }
                        coco_data["annotations"].append(annotation)
                        ann_id += 1

                # Move to next image
                image_id += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}, skipping")
                continue

        # Save COCO format data
        if args.mini:
            output_file = osp.join(
                args.out_dir, f'{args.mini_prefix}nuscenes_2d_{split}.json')
        else:
            output_file = osp.join(args.out_dir, f'nuscenes_2d_{split}.json')

        with open(output_file, 'w') as f:
            json.dump(coco_data, f)

        print(
            f"Created {split} dataset with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations.")


if __name__ == '__main__':
    main()
