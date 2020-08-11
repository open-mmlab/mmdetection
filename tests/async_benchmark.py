import asyncio
import os
import shutil
import urllib

import mmcv
import torch

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result)
from mmdet.utils.contextmanagers import concurrent
from mmdet.utils.profiling import profile_time


async def main():
    """

    Benchmark between async and synchronous inference interfaces.

    Sample runs for 20 demo images on K80 GPU, model - mask_rcnn_r50_fpn_1x:

    async	sync

    7981.79 ms	9660.82 ms
    8074.52 ms	9660.94 ms
    7976.44 ms	9406.83 ms

    Async variant takes about 0.83-0.85 of the time of the synchronous
    interface.

    """
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    config_file = os.path.join(project_dir,
                               'configs/mask_rcnn_r50_fpn_1x_coco.py')
    checkpoint_file = os.path.join(
        project_dir, 'checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth')

    if not os.path.exists(checkpoint_file):
        url = ('https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection'
               '/models/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth')
        print(f'Downloading {url} ...')
        local_filename, _ = urllib.request.urlretrieve(url)
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        shutil.move(local_filename, checkpoint_file)
        print(f'Saved as {checkpoint_file}')
    else:
        print(f'Using existing checkpoint {checkpoint_file}')

    device = 'cuda:0'
    model = init_detector(
        config_file, checkpoint=checkpoint_file, device=device)

    # queue is used for concurrent inference of multiple images
    streamqueue = asyncio.Queue()
    # queue size defines concurrency level
    streamqueue_size = 4

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))

    # test a single image and show the results
    img = mmcv.imread(os.path.join(project_dir, 'demo/demo.jpg'))

    # warmup
    await async_inference_detector(model, img)

    async def detect(img):
        async with concurrent(streamqueue):
            return await async_inference_detector(model, img)

    num_of_images = 20
    with profile_time('benchmark', 'async'):
        tasks = [
            asyncio.create_task(detect(img)) for _ in range(num_of_images)
        ]
        async_results = await asyncio.gather(*tasks)

    with torch.cuda.stream(torch.cuda.default_stream()):
        with profile_time('benchmark', 'sync'):
            sync_results = [
                inference_detector(model, img) for _ in range(num_of_images)
            ]

    result_dir = os.path.join(project_dir, 'demo')
    show_result(
        img,
        async_results[0],
        model.CLASSES,
        score_thr=0.5,
        show=False,
        out_file=os.path.join(result_dir, 'result_async.jpg'))
    show_result(
        img,
        sync_results[0],
        model.CLASSES,
        score_thr=0.5,
        show=False,
        out_file=os.path.join(result_dir, 'result_sync.jpg'))


if __name__ == '__main__':
    asyncio.run(main())
