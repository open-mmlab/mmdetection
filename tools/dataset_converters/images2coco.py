import argparse
import os

import mmcv
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert images to coco format without annotations')
    parser.add_argument('img_paths', help='image root path')
    parser.add_argument(
        'classes', type=str, help='txt file name of storage class list')
    parser.add_argument(
        'out',
        type=str,
        help='output annotation json file name, The save dir is in the same '
        'directory as img_paths')
    parser.add_argument(
        '-e',
        '--exclude-extensions',
        type=str,
        nargs='+',
        help='suffix name to be excluded. e.g., "png", "bmp"')
    args = parser.parse_args()
    return args


def collect_image_infos(path, exclude_extensions=None):
    img_infos = []

    images_generator = mmcv.scandir(path, recursive=True)
    for image_path in mmcv.track_iter_progress(list(images_generator)):
        if exclude_extensions is None or (
                exclude_extensions is not None
                and not image_path.lower().endswith(exclude_extensions)):
            image_path = os.path.join(path, image_path)
            img_pillow = Image.open(image_path)
            img_info = {
                'filename': image_path,
                'width': img_pillow.width,
                'height': img_pillow.height,
            }
            img_infos.append(img_info)
    return img_infos


def cvt_to_coco_json(img_infos, classes):
    image_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    for category_id, name in enumerate(classes):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for img_dict in img_infos:
        file_name = img_dict['filename']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(img_dict['height'])
        image_item['width'] = int(img_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        image_id += 1
    return coco


def main():
    args = parse_args()
    assert args.out.endswith(
        'json'), 'The output file name must be json suffix'

    # 1 load image list info
    img_infos = collect_image_infos(args.img_paths, args.exclude_extensions)

    # 2 convert to coco format data
    classes = mmcv.list_from_file(args.classes)
    coco_infos = cvt_to_coco_json(img_infos, classes)

    # 3 dump
    save_dir = os.path.join(args.img_paths, '..', 'annotations')
    mmcv.mkdir_or_exist(save_dir)
    save_path = os.path.join(save_dir, args.out)
    mmcv.dump(coco_infos, save_path)
    print(f'save json file: {save_path}')


if __name__ == '__main__':
    main()
