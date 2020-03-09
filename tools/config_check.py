import argparse
import os
import os.path as osp

from mmcv import Config


def collect_cfgs(folder):
    collected_files = dict()
    for root, _, files in os.walk(folder):
        if 'component' not in root:
            for file in files:
                if file.endswith('.py'):
                    file_path = osp.abspath(osp.join(root, file))
                    file_name = file_path.replace(osp.abspath(folder), '')
                    collected_files[file_name] = file_path

    return collected_files


def check(src, dst):
    src_files = collect_cfgs(src)
    dst_files = collect_cfgs(dst)

    assert set(dst_files.keys()).issubset(set(src_files.keys()))

    print(dst_files)
    for file_name, dst_path in dst_files.items():
        print('checking: {}'.format(file_name))
        src_path = src_files[file_name]
        src_dict, _ = Config._file2dict(src_path)

        dst_dict, _ = Config._file2dict(dst_path)
        if src_dict != dst_dict:
            for k in src_dict:
                if src_dict[k] != dst_dict[k]:
                    print('{} does not match'.format(k))
                    if isinstance(src_dict[k], dict):
                        for k_ in src_dict[k]:
                            if src_dict[k][k_] != dst_dict[k][k_]:
                                print('{}.{} does not match'.format(k, k_))
                                print(src_dict[k][k_])
                                print(dst_dict[k][k_])

                    print(src_dict[k])
                    print(dst_dict[k])
            for k in dst_dict:
                if src_dict[k] != dst_dict[k]:
                    print('{} does not match'.format(k))
            raise TypeError('dict does not match')
        print('{} is checked'.format(file_name))

    # for f in set(src_files.keys()) - set(dst_files.keys()):
    #     print('{} is not in dst'.format(f))

    print('{} checked, {} not checked'.format(
        len(dst_files),
        len(src_files) - len(dst_files)))


def main():
    parser = argparse.ArgumentParser(description='Compare whether configs in '
                                     'dst folder are loaded same '
                                     'as dst folder')
    parser.add_argument('src', help='src config')
    parser.add_argument('dst', help='dst config')
    args = parser.parse_args()
    check(args.src, args.dst)


if __name__ == '__main__':
    main()
