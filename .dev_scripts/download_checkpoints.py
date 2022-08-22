# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import math
import os
import os.path as osp
from multiprocessing import Pool

import torch
from mmengine.config import Config
from mmengine.utils import mkdir_or_exist


def download(url, out_file, min_bytes=math.pow(1024, 2), progress=True):
    # math.pow(1024, 2) is mean 1 MB
    assert_msg = f"Downloaded url '{url}' does not exist " \
                 f'or size is < min_bytes={min_bytes}'
    try:
        print(f'Downloading {url} to {out_file}...')
        torch.hub.download_url_to_file(url, str(out_file), progress=progress)
        assert osp.exists(
            out_file) and osp.getsize(out_file) > min_bytes, assert_msg
    except Exception as e:
        if osp.exists(out_file):
            os.remove(out_file)
        print(f'ERROR: {e}\nRe-attempting {url} to {out_file} ...')
        os.system(f"curl -L '{url}' -o '{out_file}' --retry 3 -C -"
                  )  # curl download, retry and resume on fail
    finally:
        if osp.exists(out_file) and osp.getsize(out_file) < min_bytes:
            os.remove(out_file)  # remove partial downloads

        if not osp.exists(out_file):
            print(f'ERROR: {assert_msg}\n')
        print('=========================================\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Download checkpoints')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'out', type=str, help='output dir of checkpoints to be stored')
    parser.add_argument(
        '--nproc', type=int, default=16, help='num of Processes')
    parser.add_argument(
        '--intranet',
        action='store_true',
        help='switch to internal network url')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    mkdir_or_exist(args.out)

    cfg = Config.fromfile(args.config)

    checkpoint_url_list = []
    checkpoint_out_list = []

    for model in cfg:
        model_infos = cfg[model]
        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            checkpoint = model_info['checkpoint']
            out_file = osp.join(args.out, checkpoint)
            if not osp.exists(out_file):

                url = model_info['url']
                if args.intranet is True:
                    url = url.replace('.com', '.sensetime.com')
                    url = url.replace('https', 'http')

                checkpoint_url_list.append(url)
                checkpoint_out_list.append(out_file)

    if len(checkpoint_url_list) > 0:
        pool = Pool(min(os.cpu_count(), args.nproc))
        pool.starmap(download, zip(checkpoint_url_list, checkpoint_out_list))
    else:
        print('No files to download!')
