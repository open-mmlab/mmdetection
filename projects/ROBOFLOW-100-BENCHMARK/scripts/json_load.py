import json
import argparse

parser = argparse.ArgumentParser(description='load dataset/json/config')

# 添加参数
parser.add_argument('--dataset', type=str,
                    help='present dataset')
parser.add_argument('--json', type=str,
                    help='present dataset json name')
parser.add_argument('--config', type=str,
                    help='present config')

# 解析参数
args = parser.parse_args()

def main():
    json_name = args.json
    with open(json_name) as f:
        data = json.load(f)
    classes = []
    for i in range(len(data['categories'])):
        classes.append(data['categories'][i]['name'])
    
    config = args.config
    with open(config, 'r') as f:
        lines = f.readlines()
        # 在第5行之后插入新代码
    lines[1] = f'data_root = \'{args.dataset}/\'\n' 
    lines[2] = f'classes = {classes}\n'
    lines[3] = f'classes_num = {len(classes)}\n'
    with open(config, 'w') as f:
        f.writelines(lines)
    print(len(classes))

if __name__ == "__main__":
    main()