import argparse
import os
import os.path as osp
from multiprocessing import Pool
import math

import torch

import mmcv
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Download checkpoints')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'out', type=str, help='output dir of checkpoints to be stored')
    parser.add_argument(
        '--nproc', type=int, default=48, help='num of Processes')
    args = parser.parse_args()
    return args


# refer to https://github.com/hhaAndroid/yolov5/blob/learn/utils/google_utils.py#L20
def safe_download(url, file, min_bytes=math.pow(1024, 2), progress=True):
    # math.pow(1024, 2) is mean 1MB
    assert_msg = f"Downloaded url '{url}' does not exist or size is < min_bytes={min_bytes}"
    try:
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=progress)
        assert osp.exists(file) and osp.getsize(file) > min_bytes, assert_msg  # check
    except Exception as e:
        if osp.exists(file):
            os.remove(file)
        print(f'ERROR: {e}\nRe-attempting {url} to {file} ...')
        os.system(f"curl -L '{url}' -o '{file}' --retry 3 -C -")  # curl download, retry and resume on fail
    finally:
        if osp.exists(file) and osp.getsize(file) < min_bytes:  # check
            os.remove(file)  # remove partial downloads

        if not osp.exists(file):
            print(f"ERROR: {assert_msg}\n")
        print(f"=========================================\n")


def _special_process_url(url, config, checkpoint):
    if config.find('+') >= 0:
        url = url.replace('+', '%2B')
    elif config.find('yolact') >= 0:
        url = osp.join('https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/yolact/', checkpoint)

    return url


if __name__ == '__main__':
    args = parse_args()

    url_prefix = 'https://download.openmmlab.com/mmdetection/v2.0'

    mmcv.mkdir_or_exist(args.out)

    cfg = Config.fromfile(args.config)

    url_list = []
    download_list = []

    for model_key in cfg:
        model_infos = cfg[model_key]
        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            config = model_info['config']
            config = config.replace('configs/', '').replace('.py', '')
            checkpoint = model_info['checkpoint']

            url = osp.join(url_prefix, config, checkpoint)
            url = _special_process_url(url, config, checkpoint)

            download_file = osp.join(args.out, checkpoint)
            if not osp.exists(download_file):
                url_list.append(url)
                download_list.append(download_file)

    if len(url_list) > 0:
        pool = Pool(args.nproc)
        pool.starmap(
            safe_download,
            zip(url_list, download_list))
    else:
        print(f"ERROR: No files to download!\n")
