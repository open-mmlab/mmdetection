from mmcv.engine import (default_args_parser, gather_info, setup_cfg,
                         setup_envs, setup_logger)
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import test_launch, train_launch
from mmdet.utils import collect_env, get_root_logger


def main():
    args = default_args_parser().parse_args()
    cfg = setup_cfg(args)

    distributed = setup_envs(cfg, args)
    timestamp = setup_logger(cfg, get_root_logger)
    meta = gather_info(cfg, args, distributed, get_root_logger, collect_env)

    # save mmdet version, config file content and class names in
    # checkpoints as meta data
    cfg.checkpoint_config.meta = dict(mmdet_version=__version__ +
                                      get_git_hash()[:7])

    if args.test_only:
        test_launch(
            args, cfg, distributed=distributed, timestamp=timestamp, meta=meta)
    else:
        train_launch(
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)


if __name__ == '__main__':
    main()
