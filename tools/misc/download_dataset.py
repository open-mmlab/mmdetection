import argparse
import os
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from zipfile import ZipFile

import torch


def download(url, dir, unzip=True, delete=False, curl=False, threads=1):
    def download_one(url, dir):
        f = dir / Path(url).name
        if Path(url).is_file():
            Path(url).rename(f)
        elif not f.exists():
            print('Downloading {} to {}'.format(url, f))
            if curl:
                os.system('curl -L {} -o {} --retry 9 -C -'.format(url, f))
            else:
                torch.hub.download_url_to_file(url, f, progress=True)
        if unzip and f.suffix in ('.zip', '.gz'):
            print('Unzipping {}'.format(f.name))
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')
            if delete:
                f.unlink()
    if not url:
        print("Only support coco now, it will support other dataset in the fulture!")
        return
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)

def main(args):
    path = Path(args.save_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    data2url = dict(coco2017=[
        'http://images.cocodataset.org/zips/train2017.zip',
        'http://images.cocodataset.org/zips/val2017.zip',
        'http://images.cocodataset.org/zips/test2017.zip',
        'http://images.cocodataset.org/annotations/' +
        'annotations_trainval2017.zip'
    ], lvis=[], voc2007=[], )
    download(
        data2url[args.dataset_name],
        dir=path,
        unzip=args.unzip,
        delete=args.delete,
        curl=not args.win,
        threads=args.threads)


if __name__ == '__main__':
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
        '--unzip', action='store_true', help='whether unzip dataset or not. if use "--delete" , the zipped files will be deleted')
    parser.add_argument('--delete', action='store_true', help='delete the download zipped files')
    parser.add_argument(
        '--threads', type=int, help='number of threading', default=4)
    parser.add_argument(
        '--win', action='store_true', help='use windows to download or not')
    args = parser.parse_args()
    main(args)
