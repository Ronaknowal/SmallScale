"""Analysis tools: loss curve plotting, ablation comparison tables.

This produces the figures and tables you'd put in a research write-up.

Usage:
    python -m evaluation.analysis --exp_dir ./experiments/
"""

import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict


def load_metrics(exp_dir: str) -> Dict[str, List[float]]:
    """Load metrics CSV into dict of lists."""
    path = Path(exp_dir) / "metrics.csv"
    if not path.exists():
        return {}
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in data:
                    data[k] = []
                try:
                    data[k].append(float(v))
                except (ValueError, TypeError):
                    data[k].append(v)
    return data


def plot_loss_curves(exp_dirs: List[str], labels: List[str], output_path: str):
    """Plot training loss curves for multiple experiments on one figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for exp_dir, label in zip(exp_dirs, labels):
        metrics = load_metrics(exp_dir)
        if not metrics:
            continue

        steps = [s for s in metrics.get("step", []) if isinstance(s, float)]

        # Training loss
        train_loss = metrics.get("train/loss", [])
        valid_pairs = [(s, l) for s, l in zip(steps, train_loss) if isinstance(l, float) and l > 0]
        if valid_pairs:
            s, l = zip(*valid_pairs)
            ax1.plot(s, l, label=label, alpha=0.8)

        # Validation loss
        val_loss = metrics.get("val/loss", [])
        valid_pairs = [(s, l) for s, l in zip(steps, val_loss) if isinstance(l, float) and l > 0]
        if valid_pairs:
            s, l = zip(*valid_pairs)
            ax2.plot(s, l, label=label, marker="o", markersize=3)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved loss curves to {output_path}")


def ablation_summary_table(exp_dirs: List[str], labels: List[str]) -> str:
    """Generate a markdown table comparing final metrics across ablations."""
    rows = []
    for exp_dir, label in zip(exp_dirs, labels):
        metrics = load_metrics(exp_dir)
        if not metrics:
            continue

        # Get final values
        train_losses = [l for l in metrics.get("train/loss", []) if isinstance(l, float)]
        val_losses = [l for l in metrics.get("val/loss", []) if isinstance(l, float)]
        throughputs = [t for t in metrics.get("perf/tokens_per_sec", []) if isinstance(t, float)]

        rows.append({
            "Experiment": label,
            "Final Train Loss": f"{train_losses[-1]:.4f}" if train_losses else "N/A",
            "Best Val Loss": f"{min(val_losses):.4f}" if val_losses else "N/A",
            "Avg Throughput (tok/s)": f"{np.mean(throughputs):,.0f}" if throughputs else "N/A",
        })

    if not rows:
        return "No data found."

    # Build markdown table
    headers = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")

    table = "\n".join(lines)
    print(table)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments")
    args = parser.parse_args()

    base = Path(args.exp_dir)
    if not base.exists():
        print(f"No experiments found at {base}")
        exit(1)

    exp_dirs = sorted([str(d) for d in base.iterdir() if d.is_dir()])
    labels = [Path(d).name for d in exp_dirs]

    if not exp_dirs:
        print("No experiment subdirectories found.")
        exit(1)

    print("\n=== Ablation Summary ===\n")
    ablation_summary_table(exp_dirs, labels)

    print("\n=== Generating Loss Curves ===\n")
    plot_loss_curves(exp_dirs, labels, str(base / "loss_curves.png"))
