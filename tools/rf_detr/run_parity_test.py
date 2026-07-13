from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream-python', required=True)
    parser.add_argument('--mmdet-python', required=True)
    parser.add_argument('--config', default='configs/rf_detr/rf_detr_small.py')
    parser.add_argument('--upstream-checkpoint', required=True)
    parser.add_argument('--mmdet-checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out-dir', default='reports/parity')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    upstream_npz = out_dir / 'upstream_outputs.npz'
    mmdet_npz = out_dir / 'mmdet_outputs.npz'
    subprocess.check_call([
        args.upstream_python, 'tools/rf_detr/run_upstream_inference.py',
        args.upstream_checkpoint, args.image, '--output',
        str(upstream_npz)
    ])
    subprocess.check_call([
        args.mmdet_python, 'tools/rf_detr/run_mmdet_inference.py',
        args.config, args.mmdet_checkpoint, args.image, '--output',
        str(mmdet_npz)
    ])
    return subprocess.call([
        sys.executable, 'tools/rf_detr/compare_results.py',
        str(upstream_npz),
        str(mmdet_npz),
        '--report',
        str(out_dir / 'parity_report.json')
    ])


if __name__ == '__main__':
    raise SystemExit(main())
