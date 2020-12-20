import os.path as osp
import time

from mmcv.engine import default_args_parser, gather_info, setup_cfg, setup_envs
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import test_launch, train_launch
from mmdet.utils import collect_env, get_root_logger


def mmdet_args_parser():
    parser = default_args_parser()
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    return parser


def main():
    args, cfg_opts = mmdet_args_parser().parse_known_args()
    cfg = setup_cfg(args, cfg_opts)

    setup_envs(cfg, dump_cfg=args.test_only)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    meta = gather_info(cfg, logger, collect_env())

    # save mmdet version, config file content and class names in
    # checkpoints as meta data
    cfg.checkpoint_config.meta = dict(mmdet_version=__version__ +
                                      get_git_hash()[:7])

    if args.test_only:
        test_launch(args, cfg, timestamp=timestamp, meta=meta)
    else:
        train_launch(
            cfg,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)


if __name__ == '__main__':
    main()
