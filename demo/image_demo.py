# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='图片路径',default=[r'D:\mmdetection\data\yexi\images\000001.jpg',r'D:\mmdetection\data\yexi\images\000001.jpg',r'D:\mmdetection\data\yexi\images\000001.jpg',])
    parser.add_argument('--config', help='模型配置文件',default=r'D:\mmdetection\configs\retinanet\retinanet_r50_fpn_1x_coco.py')
    parser.add_argument('--checkpoint', help='模型路径',default=r'D:\mmdetection\tools\work_dirs\retinanet_r50_fpn_1x_coco\epoch_12.pth')
    parser.add_argument('--out-file', default=r'D:\mmdetection\data\000001.jpg', help='图片输出路径')
    parser.add_argument(
        '--device', default='cuda:0', help='用于推理的设备')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='用于可视化检测结果的调色板')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='box置信度阈值')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='是否为异步推理设置异步选项.')
    args = parser.parse_args()
    return args


def main(args):
    # 从配置文件和权重文件构建模型
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # 测试单张图片
    result = inference_detector(model, args.img)
    # 显示结果
    show_result_pyplot(
        model,
        args.img,
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
