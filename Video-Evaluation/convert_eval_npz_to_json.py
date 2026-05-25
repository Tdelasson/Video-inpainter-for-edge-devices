import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def safe_mean(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(np.mean(value))
    if isinstance(value, (np.floating, np.integer, float, int)):
        return float(value)
    return None


def _normalize_name(value: Optional[str]) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _candidate_results_roots(cli_root: Optional[str]) -> List[Path]:
    script_dir = Path(__file__).resolve().parent
    roots = []
    if cli_root:
        roots.append(Path(cli_root).expanduser())

    roots.extend(
        [
            Path.cwd() / "Results",
            script_dir / "Results",
            (script_dir / "../Results"),
            (script_dir / "../nvidia_jetson/Results"),
        ]
    )

    deduped = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _load_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _find_inference_summary(
    model: Optional[str],
    dataset: Optional[str],
    mask_type: Optional[str],
    results_root: Optional[str],
) -> Optional[Dict]:
    if not model or not dataset or not mask_type:
        return None

    model_norm = _normalize_name(model)
    dataset_norm = _normalize_name(dataset)
    mask_norm = _normalize_name(mask_type)

    for root in _candidate_results_roots(results_root):
        if not root.exists():
            continue

        # Fast path: exact folder match.
        direct = root / model / dataset / mask_type / 'summary.json'
        if direct.exists():
            direct_data = _load_json(direct)
            if direct_data is not None:
                return direct_data

        # Case/format-insensitive model folder match.
        for model_dir in root.iterdir():
            if not model_dir.is_dir():
                continue
            if _normalize_name(model_dir.name) != model_norm:
                continue
            summary_path = model_dir / dataset / mask_type / 'summary.json'
            if summary_path.exists():
                summary_data = _load_json(summary_path)
                if summary_data is not None:
                    return summary_data

        # Fallback: search summaries and match values inside summary.json.
        for summary_path in root.glob('*/*/*/summary.json'):
            summary_data = _load_json(summary_path)
            if summary_data is None:
                continue
            if _normalize_name(str(summary_data.get('model'))) != model_norm:
                continue
            if _normalize_name(str(summary_data.get('dataset'))) != dataset_norm:
                continue
            if _normalize_name(str(summary_data.get('mask_type'))) != mask_norm:
                continue
            return summary_data

    return None


def main():
    parser = argparse.ArgumentParser(description='Convert evaluation .npz/.npy results to compact JSON summary')
    parser.add_argument('input_path', help='Path to the evaluation output file (.npz or .npy)')
    parser.add_argument('output_path', help='Where to save the JSON file')
    parser.add_argument('--dataset', default=None, help='Optional dataset name, e.g. DAVIS')
    parser.add_argument('--mask_type', default=None, help='Optional mask type, e.g. synthetic or RealObject')
    parser.add_argument('--model', default=None, help='Optional model name, e.g. FuseFormer_OM')
    parser.add_argument(
        '--inference-results-root',
        default=None,
        help='Optional root path for inference summaries (expects Results/<model>/<dataset>/<mask>/summary.json)',
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    data = np.load(input_path, allow_pickle=True)

    if isinstance(data, np.ndarray):
        raise ValueError('Expected an .npz evaluation archive with named metrics, not a raw .npy array')

    clip_labels = data['clip_labels'].tolist() if 'clip_labels' in data.files else []

    psnr = safe_mean(data['psnr']) if 'psnr' in data.files else None
    ssim = safe_mean(data['ssim']) if 'ssim' in data.files else None
    vfid = safe_mean(data['vfid']) if 'vfid' in data.files else None

    # Paper format uses E_warp shown as ×10^-2, so multiply raw warp error by 100.
    ewarp = safe_mean(data['warp_error']) if 'warp_error' in data.files else None
    ewarp_x1e2 = (ewarp * 100.0) if ewarp is not None else None

    payload = {
        'model': args.model,
        'dataset': args.dataset,
        'mask_type': args.mask_type,
        'num_videos': len(clip_labels),
        'psnr': round(psnr, 4) if psnr is not None else None,
        'ssim': round(ssim, 4) if ssim is not None else None,
        'vfid': round(vfid, 4) if vfid is not None else None,
        'ewarp': round(ewarp, 6) if ewarp is not None else None,
        'ewarp_x1e2': round(ewarp_x1e2, 3) if ewarp_x1e2 is not None else None,
    }

    inference_summary = _find_inference_summary(
        model=args.model,
        dataset=args.dataset,
        mask_type=args.mask_type,
        results_root=args.inference_results_root,
    )

    if inference_summary is not None:
        efficiency_keys = (
            'fp16',
            'fps',
            'latency_ms',
            'peak_memory_mb',
            'baseline_allocated_mb',
            'baseline_reserved_mb',
            'peak_allocated_mb',
            'peak_reserved_mb',
            'cuda_total_mb',
            'cuda_used_start_mb',
            'cuda_used_end_mb',
        )
        for key in efficiency_keys:
            if key in inference_summary:
                payload[key] = inference_summary[key]

        # Fill missing metadata from inference summary when CLI args are omitted.
        for key in ('model', 'dataset', 'mask_type'):
            if payload.get(key) is None and key in inference_summary:
                payload[key] = inference_summary[key]

        if payload.get('num_videos') in (None, 0) and 'num_videos' in inference_summary:
            payload['num_videos'] = inference_summary['num_videos']
    else:
        print(
            'No matching inference summary found. '
            'Quality metrics were still exported without efficiency fields.'
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f'Saved compact JSON to: {output_path}')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
