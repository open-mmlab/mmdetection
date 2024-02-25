import argparse
import os

import torch


def split_prompt(weight_path, save_path='./', real_name_list=[]):
    assert os.path.isfile(weight_path), f'权重文件路径:{weight_path}有误,不存在对应文件'
    promptmodel = torch.load(weight_path)
    weight = promptmodel['state_dict']
    prompt_weight = []
    for n, v in weight.items():
        if 'learning_prompts' in n:
            prompt_weight.append(v)
    if not real_name_list:
        real_name_list = promptmodel['meta']['dataset_meta']['classes']
    assert len(real_name_list) == len(prompt_weight), '真名数量和prompt权重数量不一致'
    for v, name in zip(prompt_weight, real_name_list):
        final = {}
        final[name] = {}
        final[name]['embeddning'] = v
        path = os.path.join(save_path, os.path.expanduser(f'{name}.pth'))
        torch.save(final, path)


def main():
    parser = argparse.ArgumentParser(
        description='Convert prompt from model weight.')
    parser.add_argument(
        '--weight_path', default='v.pth', help='src model path, type:str')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument(
        '--real_name_list',
        nargs='+',
        default=[],
        help='real_name, type:list[str]',
        required=True)
    parser.add_argument(
        '--save_path', default='./', help='save path, type:str')
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    split_prompt(args.weight_path, args.save_path, args.real_name_list)


if __name__ == '__main__':
    main()
