"""This script helps to convert yolo-style dataset to the coco format.

Usage:
    $ python yolo2coco.py /path/to/dataset # image_dir

Note:
    1. Before running this script, please make sure the root directory
    of your dataset is formatted in the following struction:
    .
    └── $ROOT_PATH
        ├── classes.txt
        ├── labels
        │    ├── a.txt
        │    ├── b.txt
        │    └── ...
        ├── images
        │    ├── a.jpg
        │    ├── b.png
        │    └── ...
        └── ...
    2. The script will automatically check whether the corresponding
    `train.txt`, ` val.txt`, and `test.txt` exist under your `image_dir`
    or not. If these files are detected, the script will organize the
    dataset. The image paths in these files must be ABSOLUTE paths.
    3. Once the script finishes, the result files will be saved in the
    directory named 'annotations' in the root directory of your dataset.
    The default output file is result.json. The root directory folder may
    look like this in the root directory after the converting:
    .
    └── $ROOT_PATH
        ├── annotations
        │    ├── result.json
        │    └── ...
        ├── classes.txt
        ├── labels
        │    ├── a.txt
        │    ├── b.txt
        │    └── ...
        ├── images
        │    ├── a.jpg
        │    ├── b.png
        │    └── ...
        └── ...
    4. After converting to coco, you can use the
    `tools/analysis_tools/browse_coco_json.py` script to visualize
    whether it is correct.
"""
import argparse
import os
import os.path as osp

import mmcv
import mmengine

IMG_EXTENSIONS = ('.jpg', '.png', '.jpeg')


def check_existence(file_path: str):
    """Check if target file is existed."""
    if not osp.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist!')


def get_image_info(yolo_image_dir, idx, file_name):
    """Retrieve image information."""
    img_path = osp.join(yolo_image_dir, file_name)
    check_existence(img_path)

    img = mmcv.imread(img_path)
    height, width = img.shape[:2]
    img_info_dict = {
        'file_name': file_name,
        'id': idx,
        'width': width,
        'height': height
    }
    return img_info_dict, height, width


def convert_bbox_info(label, idx, obj_count, image_height, image_width):
    """Convert yolo-style bbox info to the coco format."""
    label = label.strip().split()
    x = float(label[1])
    y = float(label[2])
    w = float(label[3])
    h = float(label[4])

    # convert x,y,w,h to x1,y1,x2,y2
    x1 = (x - w / 2) * image_width
    y1 = (y - h / 2) * image_height
    x2 = (x + w / 2) * image_width
    y2 = (y + h / 2) * image_height

    cls_id = int(label[0])
    width = max(0., x2 - x1)
    height = max(0., y2 - y1)
    coco_format_info = {
        'image_id': idx,
        'id': obj_count,
        'category_id': cls_id,
        'bbox': [x1, y1, width, height],
        'area': width * height,
        'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
        'iscrowd': 0
    }
    obj_count += 1
    return coco_format_info, obj_count


def organize_by_existing_files(image_dir: str, existed_categories: list):
    """Format annotations by existing train/val/test files."""
    categories = ['train', 'val', 'test']
    image_list = []

    for cat in categories:
        if cat in existed_categories:
            txt_file = osp.join(image_dir, f'{cat}.txt')
            print(f'Start to read {cat} dataset definition')
            assert osp.exists(txt_file)

            with open(txt_file) as f:
                img_paths = f.readlines()
                img_paths = [
                    os.path.split(img_path.strip())[1]
                    for img_path in img_paths
                ]  # split the absolute path
                image_list.append(img_paths)
        else:
            image_list.append([])
    return image_list[0], image_list[1], image_list[2]


def convert_yolo_to_coco(image_dir: str):
    """Convert annotations from yolo style to coco style.

    Args:
        image_dir (str): the root directory of your datasets which contains
            labels, images, classes.txt, etc
    """
    print(f'Start to load existing images and annotations from {image_dir}')
    check_existence(image_dir)

    # check local environment
    yolo_label_dir = osp.join(image_dir, 'labels')
    yolo_image_dir = osp.join(image_dir, 'images')
    yolo_class_txt = osp.join(image_dir, 'classes.txt')
    check_existence(yolo_label_dir)
    check_existence(yolo_image_dir)
    check_existence(yolo_class_txt)
    print(f'All necessary files are located at {image_dir}')

    train_txt_path = osp.join(image_dir, 'train.txt')
    val_txt_path = osp.join(image_dir, 'val.txt')
    test_txt_path = osp.join(image_dir, 'test.txt')
    existed_categories = []
    print(f'Checking if train.txt, val.txt, and test.txt are in {image_dir}')
    if osp.exists(train_txt_path):
        print('Found train.txt')
        existed_categories.append('train')
    if osp.exists(val_txt_path):
        print('Found val.txt')
        existed_categories.append('val')
    if osp.exists(test_txt_path):
        print('Found test.txt')
        existed_categories.append('test')

    # prepare the output folders
    output_folder = osp.join(image_dir, 'annotations')
    if not osp.exists(output_folder):
        os.makedirs(output_folder)
        check_existence(output_folder)

    # start the convert procedure
    with open(yolo_class_txt) as f:
        classes = f.read().strip().split()

    indices = os.listdir(yolo_image_dir)
    total = len(indices)

    dataset = {'images': [], 'annotations': [], 'categories': []}
    if existed_categories == []:
        print('These files are not located, no need to organize separately.')
        for i, cls in enumerate(classes, 0):
            dataset['categories'].append({'id': i, 'name': cls})
    else:
        print('Need to organize the data accordingly.')
        train_dataset = {'images': [], 'annotations': [], 'categories': []}
        val_dataset = {'images': [], 'annotations': [], 'categories': []}
        test_dataset = {'images': [], 'annotations': [], 'categories': []}

        # category id starts from 0
        for i, cls in enumerate(classes, 0):
            train_dataset['categories'].append({'id': i, 'name': cls})
            val_dataset['categories'].append({'id': i, 'name': cls})
            test_dataset['categories'].append({'id': i, 'name': cls})
        train_img, val_img, test_img = organize_by_existing_files(
            image_dir, existed_categories)

    obj_count = 0
    skipped = 0
    converted = 0
    for idx, image in enumerate(mmengine.track_iter_progress(indices)):
        img_info_dict, image_height, image_width = get_image_info(
            yolo_image_dir, idx, image)

        if existed_categories != []:
            if image in train_img:
                dataset = train_dataset
            elif image in val_img:
                dataset = val_dataset
            elif image in test_img:
                dataset = test_dataset

        dataset['images'].append(img_info_dict)

        img_name = osp.splitext(image)[0]
        label_path = f'{osp.join(yolo_label_dir, img_name)}.txt'
        if not osp.exists(label_path):
            # if current image is not annotated or the annotation file failed
            print(
                f'WARNING: {label_path} does not exist. Please check the file.'
            )
            skipped += 1
            continue

        with open(label_path) as f:
            labels = f.readlines()
            for label in labels:
                coco_info, obj_count = convert_bbox_info(
                    label, idx, obj_count, image_height, image_width)
                dataset['annotations'].append(coco_info)
        converted += 1

    # saving results to result json
    if existed_categories == []:
        out_file = osp.join(image_dir, 'annotations/result.json')
        print(f'Saving converted results to {out_file} ...')
        mmengine.dump(dataset, out_file)
    else:
        for category in existed_categories:
            out_file = osp.join(output_folder, f'{category}.json')
            print(f'Saving converted results to {out_file} ...')
            if category == 'train':
                mmengine.dump(train_dataset, out_file)
            elif category == 'val':
                mmengine.dump(val_dataset, out_file)
            elif category == 'test':
                mmengine.dump(test_dataset, out_file)

    # simple statistics
    print(f'Process finished! Please check at {output_folder} .')
    print(f'Number of images found: {total}, converted: {converted},',
          f'and skipped: {skipped}. Total annotation count: {obj_count}.')
    print('You can use tools/analysis_tools/browse_coco_json.py to visualize!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'image_dir',
        type=str,
        help='dataset directory with ./images and ./labels, classes.txt, etc.')
    arg = parser.parse_args()
    convert_yolo_to_coco(arg.image_dir)
