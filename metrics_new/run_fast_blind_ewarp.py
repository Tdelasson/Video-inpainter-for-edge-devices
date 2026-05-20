from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export inpainting results to fast_blind format and compute warping error"
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fast-blind-root", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="DAVIS")
    parser.add_argument("--mask-type", type=str, default="synthetic", choices=["synthetic", "RealObject"])
    parser.add_argument("--frames-subdir", type=str, default="JPEGImages_432_240")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--pred-subdir", type=str, default="_official_eval_pred")
    parser.add_argument("--task", type=str, default="inpainting")
    parser.add_argument("--eval-width", type=int, default=None)
    parser.add_argument("--eval-height", type=int, default=None)
    parser.add_argument("--copy-input", action="store_true", help="Refresh fast_blind input frames")
    parser.add_argument("--cpu", action="store_true", help="Run flow computation on CPU")
    parser.add_argument("--skip-flow", action="store_true", help="Skip compute_flow_occlusion.py")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    return parser.parse_args()


def _discover_models(results_root: Path, dataset: str, mask_type: str, pred_subdir: str) -> list[str]:
    split_dir = "synthetic" if mask_type == "synthetic" else "RealObject"
    found: list[str] = []
    for child in sorted(results_root.iterdir()):
        pred_root = child / dataset / split_dir / pred_subdir
        if pred_root.exists():
            found.append(child.name)
    return found


def _run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _write_jpg(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype(np.uint8)).save(path, quality=95)


def _resize_rgb(arr: np.ndarray, target_size: tuple[int, int] | None) -> np.ndarray:
    if target_size is None:
        return arr
    pil = Image.fromarray(arr.astype(np.uint8))
    pil = pil.resize(target_size, Image.BICUBIC)
    return np.asarray(pil, dtype=np.uint8)


def _export_input_frames(
    test_json: dict[str, int],
    dataset_root: Path,
    frames_subdir: str,
    fb_input_root: Path,
    eval_size: tuple[int, int] | None,
) -> None:
    for video_name, frame_count in test_json.items():
        for t in range(frame_count):
            src = dataset_root / frames_subdir / video_name / f"{t:05d}.jpg"
            if not src.exists():
                raise FileNotFoundError(f"Missing frame: {src}")
            dst = fb_input_root / video_name / f"{t:05d}.jpg"
            img = np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)
            img = _resize_rgb(img, eval_size)
            _write_jpg(dst, img)


def _export_model_frames(
    model_name: str,
    test_json: dict[str, int],
    pred_root: Path,
    fb_model_root: Path,
    eval_size: tuple[int, int] | None,
) -> None:
    for video_name, frame_count in test_json.items():
        for t in range(frame_count):
            src = pred_root / video_name / f"frame_{t:04d}_pred.png"
            if not src.exists():
                raise FileNotFoundError(f"Missing prediction for {model_name}: {src}")
            dst = fb_model_root / video_name / f"{t:05d}.jpg"
            img = np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)
            img = _resize_rgb(img, eval_size)
            _write_jpg(dst, img)


def _read_warp_error(metric_path: Path) -> float | None:
    if not metric_path.exists():
        return None
    vals = np.loadtxt(metric_path, dtype=np.float64)
    if vals.ndim == 0:
        return float(vals)
    return float(vals[-1])


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    results_root = repo_root / "nvidia_jetson" / "Results"
    dataset_root = repo_root / "nvidia_jetson" / "Test_Data" / args.dataset

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not args.fast_blind_root.exists():
        raise FileNotFoundError(f"fast_blind root not found: {args.fast_blind_root}")

    test_json_path = dataset_root / "test.json"
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_json = json.load(f)

    models = args.models or _discover_models(results_root, args.dataset, args.mask_type, args.pred_subdir)
    if not models:
        raise RuntimeError("No models found")

    eval_size = None
    if args.eval_width is not None and args.eval_height is not None:
        eval_size = (args.eval_width, args.eval_height)
    elif args.eval_width is not None or args.eval_height is not None:
        raise ValueError("Provide both --eval-width and --eval-height")

    fb_root = args.fast_blind_root.resolve()
    data_root = fb_root / "data" / "test"
    input_root = data_root / "input" / args.dataset
    list_dir = fb_root / "lists"
    list_dir.mkdir(parents=True, exist_ok=True)

    list_path = list_dir / f"{args.dataset}_test.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for video_name in test_json.keys():
            f.write(video_name + "\n")

    if args.copy_input:
        if input_root.exists():
            shutil.rmtree(input_root)
        _export_input_frames(test_json, dataset_root, args.frames_subdir, input_root, eval_size)

    if not args.skip_flow:
        cmd = [
            "python",
            "compute_flow_occlusion.py",
            "-dataset",
            args.dataset,
            "-phase",
            "test",
            "-data_dir",
            "data",
            "-list_dir",
            "lists",
        ]
        if args.cpu:
            cmd.append("-cpu")
        _run(cmd, cwd=fb_root)

    split_dir = "synthetic" if args.mask_type == "synthetic" else "RealObject"
    output_rows: list[dict] = []

    for model_name in models:
        pred_root = results_root / model_name / args.dataset / split_dir / args.pred_subdir
        if not pred_root.exists():
            print(f"Skipping {model_name}, missing {pred_root}")
            continue

        fb_method = model_name
        fb_model_root = data_root / fb_method / args.task / args.dataset
        if fb_model_root.exists():
            shutil.rmtree(fb_model_root)
        _export_model_frames(model_name, test_json, pred_root, fb_model_root, eval_size)

        _run(
            [
                "python",
                "evaluate_WarpError.py",
                "-method",
                fb_method,
                "-task",
                args.task,
                "-dataset",
                args.dataset,
                "-phase",
                "test",
                "-data_dir",
                "data",
                "-list_dir",
                "lists",
                "-redo",
            ],
            cwd=fb_root,
        )

        metric_path = data_root / fb_method / args.task / args.dataset / "WarpError.txt"
        ewarp = _read_warp_error(metric_path)
        output_rows.append(
            {
                "model": model_name,
                "dataset": args.dataset,
                "mask_type": args.mask_type,
                "task": args.task,
                "ewarp": round(ewarp, 6) if ewarp is not None else None,
                "ewarp_x1e2": round(ewarp * 100.0, 3) if ewarp is not None else None,
                "metric_file": str(metric_path),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "ewarp_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
