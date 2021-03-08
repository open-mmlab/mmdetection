import argparse
import copy
import os
import os.path as osp

import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tools.analysis_tools.robustness_eval import get_results

from mmdet import datasets
from mmdet.apis import multi_gpu_test, set_random_seed, single_gpu_test
from mmdet.core import eval_map
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def coco_eval_with_return(result_files,
                          result_types,
                          coco,
                          max_dets=(100, 300, 1000)):
    for res_type in result_types:
        assert res_type in ['proposal', 'bbox', 'segm', 'keypoints']

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    eval_results = {}
    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        img_ids = coco.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        if res_type == 'segm' or res_type == 'bbox':
            metric_names = [
                'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10',
                'AR100', 'ARs', 'ARm', 'ARl'
            ]
            eval_results[res_type] = {
                metric_names[i]: cocoEval.stats[i]
                for i in range(len(metric_names))
            }
        else:
            eval_results[res_type] = cocoEval.stats

    return eval_results


def voc_eval_with_return(result_file,
                         dataset,
                         iou_thr=0.5,
                         logger='print',
                         only_ap=True):
    det_results = mmcv.load(result_file)
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    mean_ap, eval_results = eval_map(
        det_results,
        annotations,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        logger=logger)

    if only_ap:
        eval_results = [{
            'ap': eval_results[i]['ap']
        } for i in range(len(eval_results))]

    return mean_ap, eval_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--corruptions',
        type=str,
        nargs='+',
        default='benchmark',
        choices=[
            'all', 'benchmark', 'noise', 'blur', 'weather', 'digital',
            'holdout', 'None', 'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow',
            'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
            'pixelate', 'jpeg_compression', 'speckle_noise', 'gaussian_blur',
            'spatter', 'saturate'
        ],
        help='corruptions')
    parser.add_argument(
        '--severities',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4, 5],
        help='corruption severity levels')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for pascal voc evaluation')
    parser.add_argument(
        '--summaries',
        type=bool,
        default=False,
        help='Print summaries for every corruption and severity')
    parser.add_argument(
        '--workers', type=int, default=32, help='workers per gpu')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--final-prints',
        type=str,
        nargs='+',
        choices=['P', 'mPC', 'rPC'],
        default='mPC',
        help='corruption benchmark metric to print at the end')
    parser.add_argument(
        '--final-prints-aggregate',
        type=str,
        choices=['all', 'benchmark'],
        default='benchmark',
        help='aggregate all results or only those for benchmark corruptions')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.show_dir, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out", "--show" or "show-dir"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.workers == 0:
        args.workers = cfg.data.workers_per_gpu

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed)

    if 'all' in args.corruptions:
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter',
            'saturate'
        ]
    elif 'benchmark' in args.corruptions:
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression'
        ]
    elif 'noise' in args.corruptions:
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise']
    elif 'blur' in args.corruptions:
        corruptions = [
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'
        ]
    elif 'weather' in args.corruptions:
        corruptions = ['snow', 'frost', 'fog', 'brightness']
    elif 'digital' in args.corruptions:
        corruptions = [
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
    elif 'holdout' in args.corruptions:
        corruptions = ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
    elif 'None' in args.corruptions:
        corruptions = ['None']
        args.severities = [0]
    else:
        corruptions = args.corruptions

    rank, _ = get_dist_info()
    aggregated_results = {}
    for corr_i, corruption in enumerate(corruptions):
        aggregated_results[corruption] = {}
        for sev_i, corruption_severity in enumerate(args.severities):
            # evaluate severity 0 (= no corruption) only once
            if corr_i > 0 and corruption_severity == 0:
                aggregated_results[corruption][0] = \
                    aggregated_results[corruptions[0]][0]
                continue

            test_data_cfg = copy.deepcopy(cfg.data.test)
            # assign corruption and severity
            if corruption_severity > 0:
                corruption_trans = dict(
                    type='Corrupt',
                    corruption=corruption,
                    severity=corruption_severity)
                # TODO: hard coded "1", we assume that the first step is
                # loading images, which needs to be fixed in the future
                test_data_cfg['pipeline'].insert(1, corruption_trans)

            # print info
            print(f'\nTesting {corruption} at severity {corruption_severity}')

            # build the dataloader
            # TODO: support multiple images per gpu
            #       (only minor changes are needed)
            dataset = build_dataset(test_data_cfg)
            data_loader = build_dataloader(
                dataset,
                samples_per_gpu=1,
                workers_per_gpu=args.workers,
                dist=distributed,
                shuffle=False)

            # build the model and load checkpoint
            cfg.model.train_cfg = None
            model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
            fp16_cfg = cfg.get('fp16', None)
            if fp16_cfg is not None:
                wrap_fp16_model(model)
            checkpoint = load_checkpoint(
                model, args.checkpoint, map_location='cpu')
            # old versions did not save class info in checkpoints,
            # this walkaround is for backward compatibility
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                model.CLASSES = dataset.CLASSES

            if not distributed:
                model = MMDataParallel(model, device_ids=[0])
                show_dir = args.show_dir
                if show_dir is not None:
                    show_dir = osp.join(show_dir, corruption)
                    show_dir = osp.join(show_dir, str(corruption_severity))
                    if not osp.exists(show_dir):
                        osp.makedirs(show_dir)
                outputs = single_gpu_test(model, data_loader, args.show,
                                          show_dir, args.show_score_thr)
            else:
                model = MMDistributedDataParallel(
                    model.cuda(),
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False)
                outputs = multi_gpu_test(model, data_loader, args.tmpdir)

            if args.out and rank == 0:
                eval_results_filename = (
                    osp.splitext(args.out)[0] + '_results' +
                    osp.splitext(args.out)[1])
                mmcv.dump(outputs, args.out)
                eval_types = args.eval
                if cfg.dataset_type == 'VOCDataset':
                    if eval_types:
                        for eval_type in eval_types:
                            if eval_type == 'bbox':
                                test_dataset = mmcv.runner.obj_from_dict(
                                    cfg.data.test, datasets)
                                logger = 'print' if args.summaries else None
                                mean_ap, eval_results = \
                                    voc_eval_with_return(
                                        args.out, test_dataset,
                                        args.iou_thr, logger)
                                aggregated_results[corruption][
                                    corruption_severity] = eval_results
                            else:
                                print('\nOnly "bbox" evaluation \
                                is supported for pascal voc')
                else:
                    if eval_types:
                        print(f'Starting evaluate {" and ".join(eval_types)}')
                        if eval_types == ['proposal_fast']:
                            result_file = args.out
                        else:
                            if not isinstance(outputs[0], dict):
                                result_files = dataset.results2json(
                                    outputs, args.out)
                            else:
                                for name in outputs[0]:
                                    print(f'\nEvaluating {name}')
                                    outputs_ = [out[name] for out in outputs]
                                    result_file = args.out
                                    + f'.{name}'
                                    result_files = dataset.results2json(
                                        outputs_, result_file)
                        eval_results = coco_eval_with_return(
                            result_files, eval_types, dataset.coco)
                        aggregated_results[corruption][
                            corruption_severity] = eval_results
                    else:
                        print('\nNo task was selected for evaluation;'
                              '\nUse --eval to select a task')

                # save results after each evaluation
                mmcv.dump(aggregated_results, eval_results_filename)

    if rank == 0:
        # print filan results
        print('\nAggregated results:')
        prints = args.final_prints
        aggregate = args.final_prints_aggregate

        if cfg.dataset_type == 'VOCDataset':
            get_results(
                eval_results_filename,
                dataset='voc',
                prints=prints,
                aggregate=aggregate)
        else:
            get_results(
                eval_results_filename,
                dataset='coco',
                prints=prints,
                aggregate=aggregate)


if __name__ == '__main__':
    main()
