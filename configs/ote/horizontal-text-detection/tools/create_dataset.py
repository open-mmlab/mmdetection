"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import json
import logging
import os

from ote.datasets.text_spotting import TextOnlyCocoAnnotation, str_to_class


def parse_args():
    """ Parses input arguments. """

    args = argparse.ArgumentParser()
    args.add_argument('--config', help='Path to dataset configuration file (json).',
                      required=True)
    args.add_argument('--root', help='Root data dir.',
                      required=True)
    args.add_argument('--output', help='Path where to save annotation (json).',
                      required=True)
    args.add_argument('--visualize', action='store_true', help='Visualize annotation.')
    args.add_argument('--shuffle', action='store_true', help='Shuffle annotation before visualization.')
    args.add_argument('--delay', type=int, default=1)
    return args.parse_args()


def main():
    """ Loads configuration file and creates dataset. """

    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    with open(args.config) as file:
        config = json.load(file)

    assert isinstance(config, list)
    ann = TextOnlyCocoAnnotation()
    for dataset in config:
        logging.info(f'Parsing {dataset["name"]}...')
        assert isinstance(dataset, dict)
        if os.path.islink(args.root):
            dataset['kwargs']['root'] = os.readlink(args.root)
        else:
            dataset['kwargs']['root'] = os.path.abspath(args.root)
        ann += str_to_class[dataset['name']](**dataset['kwargs'])()
        logging.info(f'Parsing {dataset["name"]} has been completed.')

    logging.info(f'Writing annotation to {args.output}...')
    ann.write(args.output)
    logging.info(f'Writing annotation to {args.output} has been completed.')

    ann = TextOnlyCocoAnnotation(args.output, os.path.dirname(args.output))
    if args.visualize:
        ann.visualize(put_text=True, imshow_delay=args.delay, shuffle=args.shuffle)


if __name__ == '__main__':
    main()
