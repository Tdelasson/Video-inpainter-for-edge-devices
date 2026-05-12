from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize evaluation frame folders (e.g. JPEGImages -> JPEGImages_256_256)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Test_Data"),
        help="Root containing DAVIS/ and YouTube-VOS/ directories.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["DAVIS", "YouTube-VOS"],
        help="Dataset names under data-root.",
    )
    parser.add_argument(
        "--src-subdir",
        type=str,
        default="JPEGImages",
        help="Source frame subfolder under each dataset.",
    )
    parser.add_argument(
        "--dst-prefix",
        type=str,
        default=None,
        help="Destination subfolder prefix. Defaults to src-subdir.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=str,
        default=["256x256", "512x512"],
        help="Target sizes as WIDTHxHEIGHT, e.g. 256x256 512x512.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing resized files.",
    )
    return parser.parse_args()


def parse_size(size_spec: str) -> tuple[int, int]:
    lower = size_spec.lower()
    if "x" not in lower:
        raise ValueError(f"Invalid size '{size_spec}'. Expected WIDTHxHEIGHT format.")
    w_str, h_str = lower.split("x", 1)
    width = int(w_str)
    height = int(h_str)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size '{size_spec}'. Width and height must be > 0.")
    return width, height


def iter_images(video_dir: Path) -> list[Path]:
    return sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def resize_dataset(
    data_root: Path,
    dataset: str,
    src_subdir: str,
    dst_prefix: str,
    target_sizes: list[tuple[int, int]],
    overwrite: bool,
) -> None:
    src_root = data_root / dataset / src_subdir
    if not src_root.exists():
        raise FileNotFoundError(f"Source frames directory not found: {src_root}")

    video_dirs = sorted(d for d in src_root.iterdir() if d.is_dir())
    print(f"[{dataset}] Found {len(video_dirs)} videos in {src_root}")

    for width, height in target_sizes:
        dst_subdir = f"{dst_prefix}_{width}_{height}"
        dst_root = data_root / dataset / dst_subdir
        dst_root.mkdir(parents=True, exist_ok=True)

        total_written = 0
        total_skipped = 0
        print(f"[{dataset}] Writing resized frames to {dst_root}")

        for video_dir in video_dirs:
            dst_video_dir = dst_root / video_dir.name
            dst_video_dir.mkdir(parents=True, exist_ok=True)

            for src_img_path in iter_images(video_dir):
                dst_img_path = dst_video_dir / src_img_path.name
                if dst_img_path.exists() and not overwrite:
                    total_skipped += 1
                    continue

                with Image.open(src_img_path) as img:
                    rgb = img.convert("RGB")
                    resized = rgb.resize((width, height), resample=Image.BILINEAR)
                    resized.save(dst_img_path)
                total_written += 1

        print(
            f"[{dataset}] {width}x{height}: wrote {total_written} images"
            f" ({total_skipped} skipped)."
        )


def main() -> None:
    args = parse_args()

    size_pairs = [parse_size(s) for s in args.sizes]
    dst_prefix = args.dst_prefix or args.src_subdir

    for dataset in args.datasets:
        resize_dataset(
            data_root=args.data_root,
            dataset=dataset,
            src_subdir=args.src_subdir,
            dst_prefix=dst_prefix,
            target_sizes=size_pairs,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
