# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmengine


def parse_args():
    parser = argparse.ArgumentParser(description='Override Category')
    parser.add_argument('data_root')
    return parser.parse_args()


def main():
    args = parse_args()

    ChessPieces = [{
        'id': 1,
        'name': '  ',
        'supercategory': 'pieces'
    }, {
        'id': 2,
        'name': 'black bishop',
        'supercategory': 'pieces'
    }, {
        'id': 3,
        'name': 'black king',
        'supercategory': 'pieces'
    }, {
        'id': 4,
        'name': 'black knight',
        'supercategory': 'pieces'
    }, {
        'id': 5,
        'name': 'black pawn',
        'supercategory': 'pieces'
    }, {
        'id': 6,
        'name': 'black queen',
        'supercategory': 'pieces'
    }, {
        'id': 7,
        'name': 'black rook',
        'supercategory': 'pieces'
    }, {
        'id': 8,
        'name': 'white bishop',
        'supercategory': 'pieces'
    }, {
        'id': 9,
        'name': 'white king',
        'supercategory': 'pieces'
    }, {
        'id': 10,
        'name': 'white knight',
        'supercategory': 'pieces'
    }, {
        'id': 11,
        'name': 'white pawn',
        'supercategory': 'pieces'
    }, {
        'id': 12,
        'name': 'white queen',
        'supercategory': 'pieces'
    }, {
        'id': 13,
        'name': 'white rook',
        'supercategory': 'pieces'
    }]

    _data_root = args.data_root + 'ChessPieces/Chess Pieces.v23-raw.coco/'
    json_data = mmengine.load(_data_root +
                              'valid/annotations_without_background.json')
    json_data['categories'] = ChessPieces
    mmengine.dump(json_data,
                  _data_root + 'valid/new_annotations_without_background.json')

    CottontailRabbits = [{
        'id': 1,
        'name': 'rabbit',
        'supercategory': 'Cottontail-Rabbit'
    }]

    _data_root = args.data_root + 'CottontailRabbits/'
    json_data = mmengine.load(_data_root +
                              'valid/annotations_without_background.json')
    json_data['categories'] = CottontailRabbits
    mmengine.dump(json_data,
                  _data_root + 'valid/new_annotations_without_background.json')

    NorthAmericaMushrooms = [{
        'id': 1,
        'name': 'flat mushroom',
        'supercategory': 'mushroom'
    }, {
        'id': 2,
        'name': 'yellow mushroom',
        'supercategory': 'mushroom'
    }]

    _data_root = args.data_root + 'NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/'  # noqa
    json_data = mmengine.load(_data_root +
                              'valid/annotations_without_background.json')
    json_data['categories'] = NorthAmericaMushrooms
    mmengine.dump(json_data,
                  _data_root + 'valid/new_annotations_without_background.json')


if __name__ == '__main__':
    main()
