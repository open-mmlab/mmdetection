import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from tqdm import tqdm
from mmengine.fileio import get


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Get image information")
    parser.add_argument("--image_dir", default='data/objects365v2/train/',help="path to image", type=str)
    parser.add_argument("--json_path", default='data/objects365v2/annotations/zhiyuan_objv2_train.json',help="path to json", type=str)
    parser.add_argument("--output_path", default='data/objects365v2/zhiyuan_objv2_train_info.txt', type=str)
    parser.add_argument("--max_workers", type=int, default=100)
    args = parser.parse_args()
    return args


args = parse_args()

objv2_backend_args = dict(
    backend='petrel',
    path_mapping=dict({
        './data/objects365v2/': 'yudong:s3://wangyudong/obj365_v2/',
        'data/objects365v2/': 'yudong:s3://wangyudong/obj365_v2/'
    }))

def get_image_info(line, image_dir):
    result = {
        "status": "",
        "id": None,
        "file_name": None,
        "height": None,
        "width": None,
        "channel": None,
    }

    file_name = line["file_name"]

    file_path = os.path.join(image_dir, file_name)
    if not os.path.isfile(file_path):
        result["status"] = "NOFOUND"
        print(line)
        return result
    try:
        img_bytes = get(file_path, backend_args)
        image = imfrombytes(img_bytes, flag='color')
    except Exception as e:
        result["status"] = "TRUNCATED"
        print(e, line)
        return result

    result["status"] = "SUCCESS"
    result["id"] = line["id"]
    result["file_name"] = line["file_name"]
    result["height"] = image.shape[0]
    result["width"] = image.shape[1]
    result["channel"] = image.shape[2]

    return result


def get_images_info(data, image_dir, record_file):
    with tqdm(total=len(data)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit up to `chunk_size` tasks at a time to avoid too many pending tasks.
            chunk_size = min(50000, args.max_workers * 500)
            for i in range(0, len(data), chunk_size):
                futures = [
                    executor.submit(get_image_info, line, image_dir)
                    for line in data[i : i + chunk_size]
                ]
                for future in concurrent.futures.as_completed(futures):
                    r = future.result()
                    status, image_id, file_name, height, width, channel = (
                        r["status"],
                        r["id"],
                        r["file_name"],
                        r["height"],
                        r["width"],
                        r["channel"],
                    )
                    if status == "SUCCESS":
                        record_file.write(f"{image_id} {file_name} {height} {width} {channel}\n")
                    elif status == "NOFOUND":
                        pass
                    elif status == "TRUNCATED":
                        pass
                    else:
                        assert False
                    pbar.update(1)


def main():
    print("loading", args.json_path)
    json_data = json.load(open(args.json_path, "r"))
    images = json_data["images"]

    record_file = open(args.output_path, "w", encoding="utf8")

    get_images_info(images, args.image_dir, record_file)

    record_file.close()


if __name__ == "__main__":
    main()
