import argparse
import logging

import mmcv
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from mmdet.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', type=str)
    parser.add_argument('n_clust', type=int)
    parser.add_argument('--min_box_size', help='min bbox Width and Height', nargs=2, type=int, default=(0, 0))
    args = parser.parse_args()
    return args


def main(args):

    cfg = mmcv.Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.train)
    logging.info(dataset)

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    logging.info('Collecting statistics...')
    wh_stats = []
    for data_batch in tqdm(iter(data_loader)):
        boxes = data_batch['gt_bboxes'].data[0][0].numpy()
        for box in boxes:
            w = box[2] - box[0] + 1
            h = box[3] - box[1] + 1
            if w > args.min_box_size[0] and h > args.min_box_size[1]:
                wh_stats.append((w, h))

    kmeans = KMeans(init='k-means++', n_clusters=args.n_clust, random_state=0).fit(wh_stats)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    areas = np.sqrt([c[0] * c[1] for c in centers])
    idx = np.argsort(areas)

    for i in idx:
        center = centers[i]
        logging.info('width: {:.3f}'.format(center[0]))
        logging.info('height: {:.3f}'.format(center[1]))

    widths = [centers[i][0] for i in idx]
    heights = [centers[i][1] for i in idx]
    logging.info(widths)
    logging.info(heights)


if __name__ == '__main__':
    log_format = '{levelname} {asctime} {filename}:{lineno:>4d}] {message}'
    date_format = '%d-%m-%y %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format, style='{')
    args = parse_args()
    main(args)
