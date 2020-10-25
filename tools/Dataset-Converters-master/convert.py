#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Converts datasets between different formats."""

import argparse
import os
import shutil

from dataset_converters.Converter import convert


def nop_fn(*args):
    pass


def create_symlink(source, dest):
    if os.path.isdir(dest):
        dest = os.path.join(dest, os.path.basename(source))
    os.symlink(os.path.abspath(source), dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dataset converter tool arguments:')
    parser.add_argument('-i', '--input-folder', help='path to source dataset', required=True)
    parser.add_argument('-o', '--output-folder', help='path to converted dataset', required=True)
    parser.add_argument('-I', '--input-format', help='format of input dataset', required=True)
    parser.add_argument('-O', '--output-format', help='format of converted dataset', required=True)
    parser.add_argument('-c', '--copy', help='copy images in converted dataset', action='store_true')
    parser.add_argument('-s', '--symlink', help='create symlinks for images in converted dataset', action='store_true')
    args = parser.parse_args()

    if args.copy and args.symlink:
        raise Exception('Cannot copy and create symlinks at the same time')

    copy_fn = nop_fn
    if args.copy:
        copy_fn = shutil.copy
    if args.symlink:
        copy_fn = create_symlink

    convert(args.input_folder, args.output_folder, args.input_format, args.output_format, copy_fn)
