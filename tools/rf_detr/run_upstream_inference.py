from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('image')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    from rfdetr import RFDETRSmall

    model = RFDETRSmall(pretrain_weights=args.checkpoint, device=args.device)
    detections = model.predict(args.image, threshold=0.0)
    np.savez(
        args.output,
        bboxes=np.asarray(detections.xyxy),
        scores=np.asarray(detections.confidence),
        labels=np.asarray(detections.class_id))
    print(json.dumps({'detections': len(detections)}))


if __name__ == '__main__':
    main()
