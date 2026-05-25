import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass
class ModelPoint:
    name: str
    psnr: float
    latency_ms: float
    memory_mb: float


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_point(item: dict, default_name: str) -> ModelPoint:
    name = str(item.get("model") or item.get("name") or default_name)
    psnr = float(item["psnr"])

    if "latency_ms" in item:
        latency_ms = float(item["latency_ms"])
    elif "latency" in item:
        latency_ms = float(item["latency"])
    elif "fps" in item:
        fps = float(item["fps"])
        if fps <= 0:
            raise ValueError(f"FPS must be > 0 to derive latency for '{name}'.")
        latency_ms = 1000.0 / fps
    else:
        raise KeyError(
            f"Missing latency field for '{name}'. Provide 'latency_ms'/'latency' or 'fps'."
        )

    if "memory_mb" in item:
        memory_mb = float(item["memory_mb"])
    elif "peak_memory_mb" in item:
        memory_mb = float(item["peak_memory_mb"])
    else:
        raise KeyError(
            f"Missing memory field for '{name}'. Provide 'memory_mb' or 'peak_memory_mb'."
        )

    return ModelPoint(name=name, psnr=psnr, latency_ms=latency_ms, memory_mb=memory_mb)


def load_points(custom_input: Optional[Path], eval_json_paths: Iterable[Path]) -> List[ModelPoint]:
    points: List[ModelPoint] = []

    if custom_input is not None:
        payload = _load_json(custom_input)
        if isinstance(payload, dict):
            payload = payload.get("models", [])
        if not isinstance(payload, list):
            raise ValueError("Custom input JSON must be a list or a dict with a 'models' list.")

        for idx, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Item #{idx} in custom input is not an object.")
            points.append(_to_point(item, default_name=f"Model {idx}"))

    for path in eval_json_paths:
        data = _load_json(path)
        points.append(_to_point(data, default_name=path.stem))

    if not points:
        raise ValueError(
            "No model points found. Use --input-json or --eval-json with one or more files."
        )

    return points


def _memory_color(memory_mb: float) -> str:
    if memory_mb < 100.0:
        return "#1f77b4"  # Blue
    if memory_mb < 1024.0:
        return "#ff7f0e"  # Orange
    if memory_mb < 2048.0:
        return "#7f3c8d"  # Purple
    return "#d62728"  # Red


def plot_points(
    points: List[ModelPoint],
    output_path: Path,
    title: str,
    realtime_latency_ms: float,
    size_scale: float,
    min_bubble_size: float,
    label_fontsize: float,
    legend_fontsize: float,
    dpi: int,
) -> None:
    psnr_vals = [p.psnr for p in points]
    latency_vals = [p.latency_ms for p in points]
    mem_vals = [p.memory_mb for p in points]

    min_psnr = min(psnr_vals)
    max_psnr = max(psnr_vals)
    min_latency = min(latency_vals)
    max_latency = max(max(latency_vals), realtime_latency_ms)

    x_margin = max((max_psnr - min_psnr) * 0.08, 0.5)
    y_margin = max((max_latency - min_latency) * 0.12, 1.0)

    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=dpi)

    ax.set_facecolor("#f4f4f4")

    lower_shade_start = max(0.0, min_latency - y_margin)
    ax.axhspan(
        lower_shade_start,
        realtime_latency_ms,
        facecolor="#b0b0b0",
        alpha=0.25,
        hatch="//",
    )
    ax.axhline(
        y=realtime_latency_ms,
        color="#5f5f5f",
        linestyle="--",
        linewidth=1.8,
        label=f"Real-time threshold ({realtime_latency_ms:.0f} ms)",
    )

    # In matplotlib scatter, marker size `s` is area in points^2.
    bubble_sizes = [max(min_bubble_size, m * size_scale) for m in mem_vals]
    bubble_colors = [_memory_color(m) for m in mem_vals]

    scatter = ax.scatter(
        psnr_vals,
        latency_vals,
        s=bubble_sizes,
        c=bubble_colors,
        edgecolors="#1c1c1c",
        linewidths=1.2,
        alpha=0.82,
        zorder=3,
    )

    annotation_artists = []
    for p, bubble_size in zip(points, bubble_sizes):
        radius_points = math.sqrt(bubble_size / math.pi)
        edge_offset = radius_points / math.sqrt(2.0)

        x_span = max(max_psnr - min_psnr, 1e-9)
        y_span = max(max_latency - min_latency, 1e-9)
        x_norm = (p.psnr - min_psnr) / x_span
        y_norm = (p.latency_ms - min_latency) / y_span

        # Keep labels compact: flip to left near right edge and downward near top edge.
        x_sign = -1.0 if x_norm > 0.78 else 1.0
        y_sign = -1.0 if y_norm > 0.82 else 1.0

        x_offset = x_sign * (edge_offset + 2.0)
        y_offset = y_sign * (edge_offset + 2.0)
        ha = "right" if x_sign < 0 else "left"
        va = "top" if y_sign < 0 else "bottom"

        is_viper = p.name.strip().lower() == "viper"
        label_text = "VIPER (ours)" if is_viper else p.name
        label_weight = "bold" if is_viper else "normal"

        text = ax.annotate(
            f"{label_text}\n{p.memory_mb:.1f} MB",
            xy=(p.psnr, p.latency_ms),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=label_fontsize,
            color="#121212",
            ha=ha,
            va=va,
            fontweight=label_weight,
            zorder=4,
        )
        annotation_artists.append(text)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#1f77b4",
            markeredgecolor="#1c1c1c",
            markeredgewidth=1.0,
            markersize=9,
            label="Memory < 100 MB",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#ff7f0e",
            markeredgecolor="#1c1c1c",
            markeredgewidth=1.0,
            markersize=9,
            label="Memory < 1 GB",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#7f3c8d",
            markeredgecolor="#1c1c1c",
            markeredgewidth=1.0,
            markersize=9,
            label="Memory < 2 GB",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#d62728",
            markeredgecolor="#1c1c1c",
            markeredgewidth=1.0,
            markersize=9,
            label="Memory < 6 GB",
        ),
        Line2D(
            [0],
            [0],
            color="#5f5f5f",
            linestyle="--",
            linewidth=1.6,
            label=f"Real-time threshold ({realtime_latency_ms:.0f} ms)",
        ),
    ]

    ax.legend(handles=handles, loc="best", frameon=True, facecolor="white", edgecolor="#cfcfcf")
    legend = ax.get_legend()
    if legend is not None:
        for txt in legend.get_texts():
            txt.set_fontsize(legend_fontsize)

    ax.set_xlim(min_psnr - x_margin, max_psnr + x_margin)
    ax.set_ylim(lower_shade_start, max_latency + y_margin)

    # Expand axis limits to include full bubble and text extents in display space.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_min_lim, x_max_lim = ax.get_xlim()
    y_min_lim, y_max_lim = ax.get_ylim()

    artists = [scatter, *annotation_artists]
    for artist in artists:
        bbox_display = artist.get_window_extent(renderer=renderer)
        bbox_data = bbox_display.transformed(ax.transData.inverted())
        x_min_lim = min(x_min_lim, bbox_data.x0)
        x_max_lim = max(x_max_lim, bbox_data.x1)
        y_min_lim = min(y_min_lim, bbox_data.y0)
        y_max_lim = max(y_max_lim, bbox_data.y1)

    x_pad = max((x_max_lim - x_min_lim) * 0.01, 0.05)
    y_pad = max((y_max_lim - y_min_lim) * 0.01, 0.5)
    ax.set_xlim(x_min_lim - x_pad, x_max_lim + x_pad)
    ax.set_ylim(y_min_lim - y_pad, y_max_lim + y_pad)

    ax.set_xlabel("PSNR (dB)", fontsize=16)
    ax.set_ylabel("Latency (ms)", fontsize=16)
    if title.strip():
        ax.set_title(title, fontsize=16, pad=12)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle=":", linewidth=0.9, color="#a5a5a5", alpha=0.85)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PSNR-vs-Latency bubble chart where bubble size is model memory usage."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help=(
            "Path to custom JSON input (list of objects or {\"models\": [...]}) with fields: "
            "model/name, psnr, latency_ms (or latency/fps), memory_mb (or peak_memory_mb)."
        ),
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        nargs="*",
        default=[],
        help="One or more evaluation JSON files (like ProPainter.json, E2FGVI.json).",
    )
    parser.add_argument(
        "--eval-json-dir",
        type=Path,
        default=None,
        help="Directory to auto-load all *.json files as model points.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("psnr_latency_memory_plot.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Chart title. Leave empty to omit title (default: no title).",
    )
    parser.add_argument(
        "--realtime-latency-ms",
        type=float,
        default=33.0,
        help="Real-time performance threshold in milliseconds (default: 33).",
    )
    parser.add_argument(
        "--size-scale",
        type=float,
        default=1.0,
        help="Bubble area scale factor applied to memory (default: 1.0).",
    )
    parser.add_argument(
        "--min-bubble-size",
        type=float,
        default=120.0,
        help=(
            "Minimum bubble area in points^2 for visibility (default: 120). "
            "Increase this if small-memory models are hard to see."
        ),
    )
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=17.0,
        help="Annotation font size for model labels (default: 17).",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=float,
        default=14.0,
        help="Legend font size (default: 14).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--save-svg",
        action="store_true",
        help="Also save an SVG copy suitable for LaTeX/papers.",
    )
    parser.add_argument(
        "--svg-output",
        type=Path,
        default=None,
        help="Optional SVG output path. If omitted with --save-svg, uses output path with .svg suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eval_paths: List[Path] = list(args.eval_json)
    if args.eval_json_dir is not None:
        eval_paths.extend(sorted(args.eval_json_dir.glob("*.json")))

    points = load_points(args.input_json, eval_paths)
    plot_points(
        points=points,
        output_path=args.output,
        title=args.title,
        realtime_latency_ms=args.realtime_latency_ms,
        size_scale=args.size_scale,
        min_bubble_size=args.min_bubble_size,
        label_fontsize=args.label_fontsize,
        legend_fontsize=args.legend_fontsize,
        dpi=args.dpi,
    )

    if args.save_svg:
        svg_path = args.svg_output if args.svg_output is not None else args.output.with_suffix(".svg")
        plot_points(
            points=points,
            output_path=svg_path,
            title=args.title,
            realtime_latency_ms=args.realtime_latency_ms,
            size_scale=args.size_scale,
            min_bubble_size=args.min_bubble_size,
            label_fontsize=args.label_fontsize,
            legend_fontsize=args.legend_fontsize,
            dpi=args.dpi,
        )
        print(f"Saved SVG chart to: {svg_path}")

    print(f"Saved chart to: {args.output}")
    print(f"Plotted {len(points)} models.")


if __name__ == "__main__":
    main()