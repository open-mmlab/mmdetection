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
    """解压tar文件到对应名字的文件夹，并提取所有的json combine后，删除其他保存图片."""
    # 创建文件夹并解压
    filepath = untar(tar_path, logger)
    '''将所有json融合为一个json'''
    # 获取解压后目录下所有的.json文件
    json_files = [f for f in os.listdir(filepath) if f.endswith('.json')]
    # 初始化一个空的列表来存储所有的数据
    all_data = []
    cnt = 0

    for file in json_files:
        with open(os.path.join(filepath, file), 'r') as f:
            df = json.load(f)
        cnt = cnt + 1
        # 将DataFrame转换为.json格式，并添加到all_data列表中
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
    # 如果文件是tar文件，就解压它
    if tarfile.is_tarfile(filepath):
        # 创建一个新的文件夹，和tar文件同名，但去掉后缀
        new_folder = os.path.splitext(filepath)[0]
        tar_name = os.path.basename(filepath)
        with tarfile.open(filepath) as tar:
            # 获取tar文件中的所有成员
            members = tar.getmembers()
            if not os.path.exists(new_folder):
                os.mkdir(new_folder)
            else:
                f = os.listdir(new_folder)
                # 打开tar文件，并解压到新的文件夹中
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


parser = argparse.ArgumentParser()
# parser.add_argument('-d', '--download_json_dir', type=str, default=None)
parser.add_argument('image_dir', type=str)  # grit raw directory
parser.add_argument('output_dir', type=str)  # processed grit output dir
parser.add_argument('--log_name', type=str, default='grit_processing.log')

args = parser.parse_args()


def main(args):
    logger = create_logger(args.log_name)
    # if args.download_json_dir != None:
    #     count_download_image(args.download_json_dir, logger)
    if args.image_dir is not None:
        all_file_name = [
            os.path.join(args.image_dir, file)
            for file in os.listdir(args.image_dir) if file.endswith('.tar')
        ]
        all_file_name.sort()
        func = partial(
            tar_processing, output_dir=args.output_dir, logger=logger)
        with Pool(processes=10) as pool:
            result = pool.imap(func=func, iterable=all_file_name)
            for r in result:
                print(result)


if __name__ == '__main__':
    main(args)
