# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmengine


def parse_args():
    parser = argparse.ArgumentParser(description='Override Category')
    parser.add_argument('data_root')
    return parser.parse_args()


def main():
    args = parse_args()

    backishUnderwater = [{
        'id': 1, 
        'name': 'crab', 
        'supercategory': 'animals'
    }, {'id': 2, 
        'name': 'fish', 
        'supercategory': 'animals'
    }, {'id': 3, 
        'name': 'jellyfish',
        'supercategory': 'animals'
    }, {'id': 4, 
        'name': 'shrimp', 
        'supercategory': 'animals'
    }, {'id': 5, 
        'name': 'small_fish', 
        'supercategory': 'animals'
    }, {'id': 6, 
        'name': 'starfish', 
        'supercategory': 'animals'
    }]

    _data_root = args.data_root + 'brackishUnderwater/960x540/'
    json_data = mmengine.load(_data_root +
                              'valid/annotations_without_background.json')
    json_data['categories'] = backishUnderwater
    mmengine.dump(json_data, 
                  _data_root + 'valid/new_annotations_without_background.json')

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
    
    EgoHandsGeneric = [{
        'id': 1,
        'name': 'hand',
        'supercategory': 'hands'
    }]

    _data_root = args.data_root + 'EgoHands/generic/'
    json_data = mmengine.load(_data_root + 
                              'valid/annotations_without_background.json')
    json_data['categories'] = EgoHandsGeneric
    mmengine.dump(json_data,
                  _data_root + 'valid/new_annotations_without_background.json')
    
    
    MountainDewCommercial = [{
        'id': 1,
        'name': 'bottle',
        'supercategory': 'bottles'
    }]

    _data_root = args.data_root + 'MountainDewCommercial/'
    json_data = mmengine.load(_data_root + 
                              'valid/annotations_without_background.json')
    json_data['categories'] = MountainDewCommercial
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
    
    OxfordPetsByBreed = [{
        'id': 1,
        'name': 'cat-Abyssinian',
        'supercategory': 'pets'
    }, {
        'id': 2,
        'name': 'cat-Bengal',
        'supercategory': 'pets'
    }, {
        'id': 3,
        'name': 'cat-Birman',
        'supercategory': 'pets'
    }, {
        'id': 4,
        'name': 'cat-Bombay',
        'supercategory': 'pets'
    }, {
        'id': 5,
        'name': 'cat-British_Shorthair',
        'supercategory': 'pets'
    }, {
        'id': 6,
        'name': 'cat-Egyptian_Mau',
        'supercategory': 'pets'
    }, {
        'id': 7,
        'name': 'cat-Maine_Coon',
        'supercategory': 'pets'
    }, {
        'id': 8,
        'name': 'cat-Persian',
        'supercategory': 'pets'
    }, {
        'id': 9,
        'name': 'cat-Ragdoll',
        'supercategory': 'pets'
    }, {
        'id': 10,
        'name': 'cat-Russian_Blue',
        'supercategory': 'pets'
    }, {
        'id': 11,
        'name': 'cat-Siamese',
        'supercategory': 'pets'
    }, {
        'id': 12,
        'name': 'cat-Sphynx',
        'supercategory': 'pets'
    }, {
        'id': 13, 
        'name': 'dog-american_bulldog', 
        'supercategory': 'pets'
    }, {
        'id': 14,
        'name': 'dog-american_pit_bull_terrier', 
        'supercategory': 'pets'
    }, {
        'id': 15,
        'name': 'dog-basset_hound', 
        'supercategory': 'pets'
    }, {
        'id': 16, 
        'name': 'dog-beagle',
        'supercategory': 'pets'
    }, {
        'id': 17, 
        'name': 'dog-boxer', 
        'supercategory': 'pets'
    }, {
        'id': 18, 
        'name': 'dog-chihuahua', 
        'supercategory': 'pets'
    }, {
        'id': 19, 
        'name': 'dog-english_cocker_spaniel', 
        'supercategory': 'pets'
    }, {
        'id': 20, 
        'name': 'dog-english_setter',
        'supercategory': 'pets'
    }, {
        'id': 21, 
        'name': 'dog-german_shorthaired', 
        'supercategory': 'pets'
    }, {
        'id': 22, 
        'name': 'dog-great_pyrenees', 
        'supercategory': 'pets'
    }, {
        'id': 23,
        'name': 'dog-havanese', 
        'supercategory': 'pets'
    }, {
        'id': 24, 
        'name': 'dog-japanese_chin',
        'supercategory': 'pets'
    }, {
        'id': 25, 
        'name': 'dog-keeshond', 
        'supercategory':'pets'
    }, {
        'id': 26, 
        'name': 'dog-leonberger', 
        'supercategory': 'pets'
    }, {
        'id': 27, 
        'name': 'dog-miniature_pinscher', 
        'supercategory': 'pets'
    }, {
        'id': 28, 
        'name': 'dog-newfoundland', 
        'supercategory': 'pets'
    }, {
        'id': 29, 
        'name': 'dog-pomeranian',
        'supercategory': 'pets'
    }, {
        'id': 30, 
        'name': 'dog-pug', 
        'supercategory': 'pets'
    }, {
        'id': 31, 
        'name': 'dog-saint_bernard', 
        'supercategory': 'pets'
    }, {
        'id': 32, 
        'name': 'dog-samoyed', 
        'supercategory': 'pets'
    }, {
        'id': 33, 
        'name': 'dog-scottish_terrier',
        'supercategory': 'pets'
    }, {
        'id': 34, 
        'name': 'dog-shiba_inu', 
        'supercategory': 'pets'
    }, {
        'id': 35, 
        'name': 'dog-staffordshire_bull_terrier', 
        'supercategory': 'pets'
    }, {
        'id': 36, 
        'name': 'dog-wheaten_terrier', 
        'supercategory': 'pets'
    }, {
        'id': 37, 
        'name': 'dog-yorkshire_terrier', 
        'supercategory': 'pets'
    }]

    _data_root = args.data_root + 'OxfordPets/by-breed/'
    json_data = mmengine.load(_data_root + 
                              'valid/annotations_without_background.json')
    json_data['categories'] = OxfordPetsByBreed
    mmengine.dump(json_data,
                  _data_root + 'valid/new_annotations_without_background.json')
    
    Packages = [{
        'id': 1,
        'name': 'package',
        'supercategory': 'packages'
    }]

    _data_root = args.data_root + 'Packages/Raw/'
    json_data = mmengine.load(_data_root + 
                              'valid/annotations_without_background.json')
    json_data['categories'] = Packages
    mmengine.dump(json_data, 
                  _data_root + 'valid/new_annotations_without_background.json')
    
    Pothole = [{
        'id': 1,
        'name': 'pothole',
        'superactegory': 'potholes'
    }]

    _data_root = args.data_root + 'pothole/'
    json_data = mmengine.load(_data_root +
                              'valid/annotations_without_background.json')
    json_data['categories'] = Pothole
    mmengine.dump(json_data,
                  _data_root + 'valid/new_annotations_without_background.json')
    
    Raccoon = [{
        'id': 1,
        'name': 'raccoon',
        'supercategory': 'raccoons'
    }]

    _data_root = args.data_root + 'Raccoon/Raccoon.v2-raw.coco/'
    json_data = mmengine.load(_data_root + 
                              'valid/annotations_without_background.json')
    json_data['categories'] = Raccoon
    mmengine.dump(json_data,
                  _data_root + 'valid/new_annotations_without_background.json')


if __name__ == '__main__':
    main()
