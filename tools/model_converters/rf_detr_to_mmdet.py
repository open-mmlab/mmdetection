from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import DefaultScope

from mmdet.registry import MODELS
import mmdet.models  # noqa: F401


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('model', 'state_dict'):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                if key == 'state_dict':
                    return {
                        k.removeprefix('model.'): v
                        for k, v in value.items()
                        if torch.is_tensor(v)
                    }
                return {k: v for k, v in value.items() if torch.is_tensor(v)}
    if isinstance(checkpoint, dict) and all(
            torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise TypeError('Could not find an RF-DETR tensor state dict.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('--config', default='configs/rf_detr/rf_detr_small.py')
    parser.add_argument('--report', default=None)
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--allow-missing-prefix', action='append', default=[])
    args = parser.parse_args()

    DefaultScope.get_instance('rf-detr-convert', scope_name='mmdet')
    cfg = Config.fromfile(args.config)
    if args.num_classes is not None:
        cfg.model.num_classes = args.num_classes
        cfg.model.model_config.num_classes = args.num_classes
    model = MODELS.build(cfg.model)
    expected = model.state_dict()

    source = _extract_state_dict(
        torch.load(args.src, map_location='cpu', weights_only=False))
    mapped = {}
    shape_mismatch = {}
    unexpected = []
    for key, tensor in source.items():
        target_key = f'model.{key}'
        if target_key not in expected:
            unexpected.append(key)
            continue
        if tuple(tensor.shape) != tuple(expected[target_key].shape):
            shape_mismatch[target_key] = {
                'source': list(tensor.shape),
                'target': list(expected[target_key].shape),
            }
            continue
        mapped[target_key] = tensor

    missing = sorted(set(expected) - set(mapped))
    allowed_prefixes = tuple(args.allow_missing_prefix)
    allowed_missing = [
        k for k in missing
        if allowed_prefixes and k.startswith(allowed_prefixes)
    ]
    for key in allowed_missing:
        mapped[key] = expected[key]
    important_missing = [k for k in missing if k not in allowed_missing]
    important = [
        'model.class_embed.weight', 'model.class_embed.bias',
        'model.bbox_embed.layers.0.weight'
    ]
    important_missing.extend(k for k in important if k not in mapped)
    important_missing = sorted(set(important_missing))

    report = {
        'source': str(args.src),
        'destination': str(args.dst),
        'config': args.config,
        'mapped': len(mapped),
        'missing': missing,
        'filled_from_target': allowed_missing,
        'unexpected': sorted(unexpected),
        'shape_mismatch': shape_mismatch,
        'important_missing': important_missing,
    }
    report_path = Path(args.report or str(args.dst) + '.mapping.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    if shape_mismatch or important_missing:
        raise SystemExit(
            'Checkpoint conversion failed; inspect ' + str(report_path))

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'state_dict': mapped,
            'meta': {
                'source': str(args.src),
                'converter': 'rf_detr_to_mmdet.py',
                'mapping_report': str(report_path),
            },
        }, args.dst)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
