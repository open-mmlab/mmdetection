from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('upstream')
    parser.add_argument('mmdet')
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--report', default='reports/parity/parity_report.json')
    args = parser.parse_args()

    up = np.load(args.upstream)
    mm = np.load(args.mmdet)
    report = {}
    ok = True
    for key in ('labels', 'scores', 'bboxes'):
        a, b = up[key], mm[key]
        same_shape = a.shape == b.shape
        close = same_shape and np.allclose(a, b, rtol=args.rtol, atol=args.atol)
        diff = np.abs(a - b).astype(float) if same_shape else np.asarray([])
        report[key] = {
            'upstream_shape': list(a.shape),
            'mmdet_shape': list(b.shape),
            'max_abs_diff': float(diff.max()) if diff.size else None,
            'mean_abs_diff': float(diff.mean()) if diff.size else None,
            'close': bool(close),
        }
        ok = ok and close
    from pathlib import Path
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
