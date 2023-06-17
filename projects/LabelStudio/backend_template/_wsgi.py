# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import logging
import logging.config
import os

logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'standard': {
            'format':
            '[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s'  # noqa E501
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'stream': 'ext://sys.stdout',
            'formatter': 'standard'
        }
    },
    'root': {
        'level': 'ERROR',
        'handlers': ['console'],
        'propagate': True
    }
})

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


def get_kwargs_from_config(config_path=_DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        return dict()
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return config


if __name__ == '__main__':

    from label_studio_ml.api import init_app

    from projects.LabelStudio.backend_template.mmdetection import MMDetection

    parser = argparse.ArgumentParser(description='Label studio')
    parser.add_argument(
        '-p',
        '--port',
        dest='port',
        type=int,
        default=9090,
        help='Server port')
    parser.add_argument(
        '--host', dest='host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument(
        '--kwargs',
        '--with',
        dest='kwargs',
        metavar='KEY=VAL',
        nargs='+',
        type=lambda kv: kv.split('='),
        help='Additional LabelStudioMLBase model initialization kwargs')
    parser.add_argument(
        '-d',
        '--debug',
        dest='debug',
        action='store_true',
        help='Switch debug mode')
    parser.add_argument(
        '--log-level',
        dest='log_level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=None,
        help='Logging level')
    parser.add_argument(
        '--model-dir',
        dest='model_dir',
        default=os.path.dirname(__file__),
        help='Directory models are store',
    )
    parser.add_argument(
        '--check',
        dest='check',
        action='store_true',
        help='Validate model instance before launching server')

    args = parser.parse_args()

    # setup logging level
    if args.log_level:
        logging.root.setLevel(args.log_level)

    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def parse_kwargs():
        param = dict()
        for k, v in args.kwargs:
            if v.isdigit():
                param[k] = int(v)
            elif v == 'True' or v == 'true':
                param[k] = True
            elif v == 'False' or v == 'False':
                param[k] = False
            elif isfloat(v):
                param[k] = float(v)
            else:
                param[k] = v
        return param

    kwargs = get_kwargs_from_config()

    if args.kwargs:
        kwargs.update(parse_kwargs())

    if args.check:
        print('Check "' + MMDetection.__name__ + '" instance creation..')
        model = MMDetection(**kwargs)

    app = init_app(
        model_class=MMDetection,
        model_dir=os.environ.get('MODEL_DIR', args.model_dir),
        redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=os.environ.get('REDIS_PORT', 6379),
        **kwargs)

    app.run(host=args.host, port=args.port, debug=args.debug)

else:
    # for uWSGI use
    app = init_app(
        model_class=MMDetection,
        model_dir=os.environ.get('MODEL_DIR', os.path.dirname(__file__)),
        redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=os.environ.get('REDIS_PORT', 6379))
