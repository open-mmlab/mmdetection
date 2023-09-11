import re
import argparse
import os
import glob
import csv

parser = argparse.ArgumentParser(description="log_name")
# parser.add_argument("result", type='str', help="present result address")
parser.add_argument("dataset", type=str, help="present dataset result address")
parser.add_argument("--result_csv", type=str, default=None, required = False, help='flag of csv')
args = parser.parse_args()

def main():
    # import pdb;pdb.set_trace()
    dataset = args.dataset.split('/')[-1]
    # args.dataset = 'projects/ROBOFLOW-100-BENCHMARK/runs/faster-RCNN-benchmark/results/'+dataset
    # print(os.listdir(args.dataset))
    dirs = [args.dataset+'/'+d for d in os.listdir(args.dataset) if os.path.isdir(args.dataset+'/'+d)]
    # print(dirs)
    dirs.sort(key=os.path.getmtime)
    # print(dirs)
    latest_dir = dirs[-1]
    latest_log_name = latest_dir.split('/')[-1]
    print(latest_log_name)
    # latest_dir = args.dataset+f'/{latest}'
    latest_log = latest_dir+f'/{latest_log_name}.log'
    print(latest_log)
    with open(latest_log, 'r') as f:
        log = f.read()

    match = re.findall(r'The best checkpoint with ([\d.]+) coco/bbox_mAP', log)
    if match:
        # value = match.group()
        # print(value)
        print(match)
        best=match[-1]
        print(best)
        key_value = [dataset, best]
        if args.result_csv==None:
            result_csv = os.path.dirname(os.path.dirname(args.dataset))+f'/{latest_log_name}_final_eval.csv'
            print(result_csv)
            with open(result_csv, mode='a') as f:
                writer = csv.writer(f)
                writer.writerow(key_value)
        else:
            print(args.result_csv)
            with open(args.result_csv, mode='a') as f:
                writer = csv.writer(f)
                writer.writerow(key_value)


    else:
        value = None
    # try:
    #     result_csv
    #     print('create result_csv')
    # except NameError:
    #     print('result_csv exists')

if __name__ == "__main__":
    main()