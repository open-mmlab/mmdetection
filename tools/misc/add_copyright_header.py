# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import os
import os.path as osp

# HEADER = '# Copyright (c) OpenMMLab, Inc. and its affiliates. \n'
HEADER = 'Copyright (c) OpenMMLab. All rights reserved.\n'

HEADER_KEYWORDS = {'Copyright', 'License'}


def contains_header(lines):
    for line in lines:
        if len(HEADER_KEYWORDS.intersection(set(line.split(' ')))):
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description='Add header to all files')
    parser.add_argument('src', type=str, help='source files to add header')
    parser.add_argument('--exclude', type=str, help='exclude folder')
    parser.add_argument(
        '--suffix',
        type=str,
        default='.py',
        help='header will be added to files with suffix')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.suffix == '.py':
        comment_symbol = '# '
    elif args.suffix == '.h' \
            or args.suffix == '.cpp' \
            or args.suffix == '.cu' \
            or args.suffix == '.cuh':
        comment_symbol = '// '
    else:
        raise ValueError
    filepath_list = []
    for root, dirs, files in os.walk(args.src):
        if args.exclude is not None and root.startswith(args.exclude):
            continue
        for file in files:
            if file.endswith(args.suffix):
                # file_path = osp.relpath(osp.join(root, file), args.src)
                file_path = osp.join(root, file)
                filepath_list.append(file_path)
    for filepath in filepath_list:
        print(f'reading: {filepath}')
        with open(filepath, 'r') as f:
            lines = f.readlines()
        if not contains_header(lines):
            print(filepath)
            with open(filepath, 'w') as f:
                f.writelines([comment_symbol + HEADER] + lines)


if __name__ == '__main__':
    main()
