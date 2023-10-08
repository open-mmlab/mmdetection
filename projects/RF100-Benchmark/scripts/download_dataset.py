from argparse import ArgumentParser
from os import environ
from pathlib import Path

from roboflow import Roboflow


def main():
    # construct the argument parser and parse the arguments
    parser = ArgumentParser()

    parser.add_argument(
        '-p',
        '--project',
        required=True,
        type=str,
        help='The project ID of the dataset found in the dataset URL.',
    )
    parser.add_argument(
        '-v',
        '--version',
        required=True,
        type=int,
        help='The version the dataset you want to use',
    )
    parser.add_argument(
        '-f',
        '--model_format',
        required=False,
        type=str,
        default='coco',
        help='The format of the export you want to use (i.e. coco or yolov5)',
    )

    parser.add_argument(
        '-l',
        '--location',
        required=False,
        type=str,
        default='./rf100',
        help='Where to store the dataset',
    )
    # parses command line arguments
    args = vars(parser.parse_args())

    try:
        api_key = environ['ROBOFLOW_API_KEY']
    except KeyError:
        raise KeyError('You must export your Roboflow api key, '
                       'to obtain one see https://docs.roboflow.com/rest-api.')
    # create location if it doesn't exist
    out_dir = Path(args['location']) / args['project']
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f'Storing {args["project"] } in {out_dir} for {args["model_format"]}')
    # get and download the dataset
    rf = Roboflow(api_key=api_key)
    project = rf.workspace('roboflow-100').project(args['project'])
    project.version(args['version']).download(
        args['model_format'], location=str(out_dir))
    print('Done!')


if __name__ == '__main__':
    main()
