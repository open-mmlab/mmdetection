import argparse
import tarfile
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile

import torch
from mmengine.utils.path import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download datasets for training')
    parser.add_argument(
        '--dataset-name', type=str, help='dataset name', default='coco2017')
    parser.add_argument(
        '--save-dir',
        type=str,
        help='the dir to save dataset',
        default='data/coco')
    parser.add_argument(
        '--unzip',
        action='store_true',
        help='whether unzip dataset or not, zipped files will be saved')
    parser.add_argument(
        '--delete',
        action='store_true',
        help='delete the download zipped files')
    parser.add_argument(
        '--threads', type=int, help='number of threading', default=4)
    args = parser.parse_args()
    return args


def download(url, dir, unzip=True, delete=False, threads=1):

    def download_one(url, dir):
        f = dir / Path(url).name
        if Path(url).is_file():
            Path(url).rename(f)
        elif not f.exists():
            print(f'Downloading {url} to {f}')
            torch.hub.download_url_to_file(url, f, progress=True)
        if unzip and f.suffix in ('.zip', '.tar'):
            print(f'Unzipping {f.name}')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)
            elif f.suffix == '.tar':
                TarFile(f).extractall(path=dir)
            if delete:
                f.unlink()
                print(f'Delete {f}')

    dir = Path(dir)
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def download_objects365v2(url, dir, unzip=True, delete=False, threads=1):

    def download_single(url, dir):

        if 'train' in url:
            saving_dir = dir / Path('train_zip')
            mkdir_or_exist(saving_dir)
            f = saving_dir / Path(url).name

            unzip_dir = dir / Path('train')
            mkdir_or_exist(unzip_dir)
        elif 'val' in url:
            saving_dir = dir / Path('val')
            mkdir_or_exist(saving_dir)
            f = saving_dir / Path(url).name

            unzip_dir = dir / Path('val')
            mkdir_or_exist(unzip_dir)
        else:
            raise NotImplementedError

        if Path(url).is_file():
            Path(url).rename(f)
        elif not f.exists():
            print(f'Downloading {url} to {f}')
            torch.hub.download_url_to_file(url, f, progress=True)

        if unzip and str(f).endswith('.tar.gz'):
            print(f'Unzipping {f.name}')
            tar = tarfile.open(f)
            tar.extractall(path=unzip_dir)
            if delete:
                f.unlink()
                print(f'Delete {f}')

    # process annotations
    full_url = []
    for _url in url:
        if 'zhiyuan_objv2_train.tar.gz' in _url or \
                'zhiyuan_objv2_val.json' in _url:
            full_url.append(_url)
        elif 'train' in _url:
            for i in range(51):
                full_url.append(f'{_url}patch{i}.tar.gz')
        elif 'val/images/v1' in _url:
            for i in range(16):
                full_url.append(f'{_url}patch{i}.tar.gz')
        elif 'val/images/v2' in _url:
            for i in range(16, 44):
                full_url.append(f'{_url}patch{i}.tar.gz')
        else:
            raise NotImplementedError

    dir = Path(dir)
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_single(*x), zip(full_url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        for u in full_url:
            download_single(u, dir)


def main():
    args = parse_args()
    path = Path(args.save_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    data2url = dict(
        # TODO: Support for downloading Panoptic Segmentation of COCO
        coco2017=[
            'http://images.cocodataset.org/zips/train2017.zip',
            'http://images.cocodataset.org/zips/val2017.zip',
            'http://images.cocodataset.org/zips/test2017.zip',
            'http://images.cocodataset.org/zips/unlabeled2017.zip',
            'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',  # noqa
            'http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip',  # noqa
            'http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip',  # noqa
            'http://images.cocodataset.org/annotations/image_info_test2017.zip',  # noqa
            'http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip',  # noqa
        ],
        lvis=[
            'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip',  # noqa
            'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip',  # noqa
        ],
        voc2007=[
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',  # noqa
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',  # noqa
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar',  # noqa
        ],
        # Note: There is no download link for Objects365-V1 right now. If you
        # would like to download Objects365-V1, please visit
        # http://www.objects365.org/ to concat the author.
        objects365v2=[
            # training annotations
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/zhiyuan_objv2_train.tar.gz',  # noqa
            # validation annotations
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/zhiyuan_objv2_val.json',  # noqa
            # training url root
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/',  # noqa
            # validation url root_1
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/images/v1/',  # noqa
            # validation url root_2
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/images/v2/'  # noqa
        ])
    url = data2url.get(args.dataset_name, None)
    if url is None:
        print('Only support COCO, VOC, LVIS, and Objects365v2 now!')
        return
    if args.dataset_name == 'objects365v2':
        download_objects365v2(
            url,
            dir=path,
            unzip=args.unzip,
            delete=args.delete,
            threads=args.threads)
    else:
        download(
            url,
            dir=path,
            unzip=args.unzip,
            delete=args.delete,
            threads=args.threads)


if __name__ == '__main__':
    main()
