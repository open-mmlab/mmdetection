# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import json
import os
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='图片路径', default=r"D:\data\beach_max\val\C3_S20220702140000_E20220702140959_320_max.jpg")
    parser.add_argument('--config', help='模型配置文件', default=r'D:\mmdetection\configs\solov2\solov2_r50_fpn_3x_coco.py')
    parser.add_argument('--checkpoint', help='模型路径',default=r"D:\mmdetection\tools\work_dirs\solov2_r50_fpn_3x_coco\epoch_36.pth")
    parser.add_argument('--out-file', default=r'D:\data\000003.jpg', help='图片输出路径')
    parser.add_argument(
        '--device', default='cpu', help='用于推理的设备')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='用于可视化检测结果的调色板,每个数据集都有它默认的调色板.或者也可以随机生成')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='box置信度阈值')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='是否为异步推理设置异步选项.')
    args = parser.parse_args()
    return args


def main(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # load_names = json.load(open(r'D:\data\beach_max\annotations\val.json','r'))
    # for img_info in load_names['images']:
    #     args.img = r'D:\data\beach_max\images' + os.sep + img_info['file_name']
    #     args.out_file = r'D:\data\beach_max\val_out' + os.sep + img_info['file_name']
    # 从配置文件和权重文件构建模型
    # 测试单张图片  理论上是支持多张图片一起推理,但是后处理阶段仅仅支持img为numpy型数据或单个路径
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
