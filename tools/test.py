# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('config', help='测试配置文件路径')
    parser.add_argument('checkpoint', help='模型文件')
    parser.add_argument('--work-dir', help='保存包含评估指标的文件的目录')
    parser.add_argument('--out', help='以pickle格式输出结果文件')
    parser.add_argument('--fuse-conv-bn', action='store_true', help='是否融合conv和bn, 这将略微提高推理速度')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU的索引(仅适用于非分布式训练)')
    parser.add_argument('--format-only', action='store_true', help='格式化输出结果而不执行评估. '
                        '当您想将结果格式化为特定格式并将其提交到测试服务器时,它很有用')
    parser.add_argument('--eval', type=str, nargs='+', help='评估指标, 取决于数据集.'
                        '例如, COCO: "bbox", "segm", "proposal", VOC: "mAP", "recall"')
    parser.add_argument('--show', action='store_true', help='是否显示结果')
    parser.add_argument('--show-dir', help='将保存绘制图像的目录')
    parser.add_argument('--show-score-thr', type=float, default=0.3, help='分数阈值(默认值：0.3)')
    parser.add_argument('--gpu-collect', action='store_true', help='是否使用gpu收集结果.')
    parser.add_argument('--tmpdir', help='临时目录，用于从多个 workers 收集结果,在未指定 gpu-collect 时可用')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖配置中的一些设置, 通过键值对的方式'
                        'xxx=yyy. 如果被覆盖的值是一个列表,它应该像 key="[a,b]" 或 key=a,b 格式'
                        '它也允许嵌套的list tuple值,例如key="[(a,b),(c,d)]" 注意引号是必须的,不能有空格.')
    parser.add_argument('--eval-options', nargs='+', action=DictAction,
                        help='用于评估的自定义选项, xxx=yyy格式的键值对将是dataset.evaluate()方法的参数')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='任务启动方式')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('请使用以下参数 "--out", "--eval", "--format-only", "--show" or "--show-dir"'
         '指定至少一项操作 (save/eval/format/show the results/save the results) ')

    if args.eval and args.format_only:
        raise ValueError('--eval 和 --format_only 不能同时指定')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('输出文件必须是pkl格式.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # 更新cfg.data_root,如果MMDET_DATASETS存在环境变量中
    update_data_root(cfg)

    if args.cfg_options is not None:  # 更新cfg_options中的配置信息到cfg中去
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)  # 该函数会新建一些参数以保持配置的兼容性.

    # 设置多进程配置
    setup_multi_processes(cfg)

    # 设置 cudnn_benchmark 在那些输入固定的模型中(比如SSD300),开启该参数后训练会更快
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 这里pretrained参数设置为None是因为后面需要加载测试模型参数,没有必要加载初始化模型
    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    cfg.gpu_ids = [args.gpu_id]  # 在非分布式测试中只支持单GPU模式
    cfg.device = get_device()
    # 首先初始化分布式环境, 因为 logger 会依赖分布式信息.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # 如果测试数据集是多个合并在一起时
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # 在bs>1时,将测试配置中'ImageToTensor'替换为'DefaultFormatBundle',
            # 主要是考虑到需要将多张图片Collect同一尺寸,因为bs=1时不需要考虑这个
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # 构建模型并加载权重
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # 旧版本框架在训练过程中没有在权重文件中保存类信息,这里是为了兼容旧版本
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        outputs = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
            or cfg.evaluation.get('gpu_collect', False))

    # 返回当前显卡索引,总卡数
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\n正在将测试结果写入 {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()
