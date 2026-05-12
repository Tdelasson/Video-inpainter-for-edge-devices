import re
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def parse_logs(file_path, phase_name):
    data = []
    # Regex to match your specific log format
    # Matches: [PhaseName] Iter X | Total: X | Mask: X ...
    pattern = re.compile(
        rf"\[{phase_name}\]\s+Iter\s+(?P<iter>\d+)\s+\|\s+"
        r"Total:\s+(?P<total>[\d.]+)\s+\|\s+"
        r"Mask:\s+(?P<mask>[\d.]+)\s+\|\s+"
        r"Frame:\s+(?P<frame>[\d.]+)\s+\|\s+"
        r"Perc:\s+(?P<perc>[\d.]+)\s+\|\s+"
        r"Style:\s+(?P<style>[\d.]+)\s+\|\s+"
        r"Temp:\s+(?P<temp>[\d.]+)\s+\|\s+"
        r"Adv:\s+(?P<adv>[\d.]+)"
    )

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                entry = {k: float(v) for k, v in match.groupdict().items()}
                data.append(entry)

    if not data:
        print(f"Error: No data found for phase '{phase_name}' in {file_path}")
        return None

    return pd.DataFrame(data).set_index('iter')


def plot_losses(df, phase_name, log_scale=False, smooth_window=50):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Top Plot: Total Loss ---
    ax1.plot(df.index, df['total'], alpha=0.3, color='gray', label='Raw Total')
    ax1.plot(df.index, df['total'].rolling(window=smooth_window).mean(),
             color='black', linewidth=2, label=f'Total (SMA {smooth_window})')

    ax1.set_title(f"Training Progress: {phase_name}", fontsize=14)
    ax1.set_ylabel("Total Loss")
    ax1.legend()
    if log_scale: ax1.set_yscale('log')

    # --- Bottom Plot: Components ---
    components = ['mask', 'frame', 'perc', 'style', 'temp', 'adv']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for comp, color in zip(components, colors):
        if df[comp].sum() > 0:  # Only plot if the component was active
            # Plot the smoothed version for clarity
            smoothed = df[comp].rolling(window=smooth_window).mean()
            ax2.plot(df.index, smoothed, label=comp.capitalize(), color=color, linewidth=1.5)

    ax2.set_ylabel("Component Loss")
    ax2.set_xlabel("Iterations")
    ax2.legend(loc='upper right', ncol=3)
    if log_scale: ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"{phase_name}_loss.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Video Inpainter Loss")
    parser.add_argument("file", type=str, help="Path to the .out or .log file")
    parser.add_argument("--phase", type=str, required=True, help="Phase name (e.g., Phase5_GAN)")
    parser.add_argument("--log", action="store_true", help="Use log scale for Y axis")
    parser.add_argument("--smooth", type=int, default=50, help="Smoothing window size")

    args = parser.parse_args()

    df = parse_logs(args.file, args.phase)
    if df is not None:
        plot_losses(df, args.phase, log_scale=args.log, smooth_window=args.smooth)