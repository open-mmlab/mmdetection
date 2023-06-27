import argparse
import json
import os.path as osp


def json_convert(path):
    with open(path, 'r+') as f:
        coco_data = json.load(f)
        coco_data['categories'].append({'id': 0, 'name': 'background'})
        coco_data['categories'] = sorted(
            coco_data['categories'], key=lambda x: x['id'])
        f.seek(0)
        json.dump(coco_data, f)
        f.truncate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert iSAID dataset to mmdetection format')
    parser.add_argument('dataset_path', help='iSAID folder path')

    args = parser.parse_args()
    dataset_path = args.dataset_path
    json_list = ['train', 'val']
    for dataset_mode in ['train', 'val']:
        json_file = 'instancesonly_filtered_' + dataset_mode + '.json'
        json_file_path = osp.join(dataset_path, dataset_mode, json_file)
        assert osp.exists(json_file_path), f'train is not in {dataset_path}'
        json_convert(json_file_path)
