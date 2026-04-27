import argparse
import json
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description='Convert evaluation .npz/.npy results to compact JSON summary')
    parser.add_argument('input_path', help='Path to the evaluation output file (.npz or .npy)')
    parser.add_argument('output_path', help='Where to save the JSON file')
    parser.add_argument('--dataset', default=None, help='Optional dataset name, e.g. DAVIS')
    parser.add_argument('--mask_type', default=None, help='Optional mask type, e.g. synthetic or RealObject')
    parser.add_argument('--model', default=None, help='Optional model name, e.g. FuseFormer_OM')
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f'Saved compact JSON to: {output_path}')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
