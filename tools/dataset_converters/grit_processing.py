import argparse
import json
import logging
import os
import tarfile
from functools import partial
from multiprocessing import Pool


def create_logger(output_file):
    logger = logging.getLogger('grit_logger')
    logger.setLevel(logging.INFO)  # set logger output level
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    fh = logging.FileHandler(output_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(console)

    return logger


def count_download_image(download_json_dir, logger):
    parquet_files = [
        f for f in os.listdir(download_json_dir) if f.endswith('.json')
    ]
    len = 0

    for file in parquet_files:
        with open(os.path.join(download_json_dir, file), 'r') as f:
            data = json.load(f)
            len = len + int(data['successes'])
        logger.info(file + 'has ' + str(data['successes']) +
                    ' successful images')

    logger.info('all files finished.', str(len),
                'images have been successfully downloaded.')


def tar_processing(tar_path, output_dir, logger):
    filepath = untar(tar_path, logger)
    json_files = [f for f in os.listdir(filepath) if f.endswith('.json')]
    all_data = []
    cnt = 0

    for file in json_files:
        with open(os.path.join(filepath, file), 'r') as f:
            df = json.load(f)
        cnt = cnt + 1
        all_data.extend([df])
    dir_name = os.path.basename(filepath)
    # write all data to a json file
    logger.info(f'{dir_name} has {cnt} jsons')
    json_name = os.path.basename(filepath) + '.json'
    if not os.path.exists(os.path.join(output_dir, 'annotations')):
        os.mkdir(os.path.join(output_dir, 'annotations'))
    with open(os.path.join(output_dir, 'annotations', json_name), 'w') as f:
        json.dump(all_data, f)
    logger.info(f'{dir_name} completed')
    cp_rm(filepath, output_dir)
    return os.path.basename(filepath)


def untar(filepath, logger):
    if tarfile.is_tarfile(filepath):
        new_folder = os.path.splitext(filepath)[0]
        tar_name = os.path.basename(filepath)
        with tarfile.open(filepath) as tar:
            members = tar.getmembers()
            if not os.path.exists(new_folder):
                os.mkdir(new_folder)
            else:
                f = os.listdir(new_folder)
                if len(members) == len(f):
                    logger.info(f'{tar_name} already decompressed')
                    return new_folder
            logger.info(f'{tar_name} decompressing...')
            os.system(f'tar -xf {filepath} -C {new_folder}')
            logger.info(f'{tar_name} decompressed!')
        return new_folder


def cp_rm(filepath, output_dir):
    # delete txt/json
    for file in os.listdir(filepath):
        if file.endswith('.txt') or file.endswith('.json'):
            os.remove(os.path.join(filepath, file))
    # move images to output dir
    target_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(os.path.join(output_dir, 'images')):
        os.mkdir(os.path.join(output_dir, 'images'))
    os.system('mv -f {} {}'.format(filepath, target_dir))


def main(args):
    logger = create_logger(args.log_name)
    all_file_name = [
        os.path.join(args.image_dir, file)
        for file in os.listdir(args.image_dir) if file.endswith('.tar')
    ]
    all_file_name.sort()
    func = partial(tar_processing, output_dir=args.output_dir, logger=logger)
    with Pool(processes=args.num_process) as pool:
        result = pool.imap(func=func, iterable=all_file_name)  # noqa
        # print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str)  # grit raw directory
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--num-process', default=10)
    parser.add_argument('--log-name', type=str, default='grit_processing.log')
    args = parser.parse_args()

    main(args)
