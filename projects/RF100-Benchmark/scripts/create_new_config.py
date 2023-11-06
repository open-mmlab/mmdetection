from argparse import ArgumentParser

from mmengine.fileio import load
from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = ArgumentParser(description='create new config')
    parser.add_argument('config')
    parser.add_argument('dataset')
    parser.add_argument('--save-dir', default='temp_configs')
    parser.add_argument('--name-json', default='scripts/labels_names.json')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = args.config
    labels_names_json = args.name_json

    mkdir_or_exist(args.save_dir)

    json_data = load(labels_names_json)
    dataset_name = [j['name'] for j in json_data]
    classes_name = [tuple(j['classes'].keys()) for j in json_data]
    if args.dataset in dataset_name:
        classes_name = classes_name[dataset_name.index(args.dataset)]
        with open(config, 'r') as file:
            content = file.read()
        new_content = content.replace("('profile_info', )", str(classes_name))
        new_content = new_content.replace('tweeter-profile', args.dataset)

        with open(f'{args.save_dir}/{args.dataset}.py', 'w') as file:
            file.write(new_content)
    else:
        raise ValueError('dataset name not found in labels_names.json')


if __name__ == '__main__':
    main()
