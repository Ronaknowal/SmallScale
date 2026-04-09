"""Analysis tools: per-suite loss curves, ranked ablation tables, best/worst summary.

Produces publication-ready figures grouped by ablation suite, plus a
ranked summary table showing which config won each comparison.

Usage:
    python -m evaluation.analysis --exp_dir ./experiments/
"""

import csv
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Map experiment name prefixes to ablation suites for grouped plotting
SUITES = {
    "Positional Encoding": ["pos_rope", "pos_alibi", "pos_learned"],
    "Attention Mechanism": ["attn_mha", "attn_gqa4", "attn_gqa2", "attn_mqa"],
    "Activation Function": ["act_swiglu", "act_gelu"],
    "Normalization": ["norm_pre_rmsnorm", "norm_pre_layernorm", "norm_post_rmsnorm", "norm_post_layernorm"],
    "Warmup Steps": ["warmup_100", "warmup_500", "warmup_1000", "warmup_2000"],
    "Learning Rate": ["lr_1e4", "lr_3e4", "lr_6e4", "lr_1e3"],
}

# Colors for cleaner plots
COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]


def load_metrics(exp_dir: str) -> Dict[str, List]:
    """Load metrics CSV into dict of lists, handling sparse rows."""
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
                if v == "":
                    data[k].append(None)
                else:
                    try:
                        data[k].append(float(v))
                    except (ValueError, TypeError):
                        data[k].append(v)
    return data


def get_metric_pairs(metrics: Dict, x_key: str, y_key: str) -> Tuple[List[float], List[float]]:
    """Extract aligned (x, y) pairs where both are valid floats."""
    xs = metrics.get(x_key, [])
    ys = metrics.get(y_key, [])
    pairs = [(x, y) for x, y in zip(xs, ys)
             if isinstance(x, float) and isinstance(y, float) and x is not None and y is not None]
    if not pairs:
        return [], []
    return zip(*pairs)


def plot_suite(suite_name: str, exp_names: List[str], base_dir: Path, output_path: Path):
    """Plot training + validation loss for one ablation suite."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Ablation: {suite_name}", fontsize=14, fontweight="bold")

    has_val = False
    for i, name in enumerate(exp_names):
        exp_dir = str(base_dir / name)
        metrics = load_metrics(exp_dir)
        if not metrics:
            continue
        color = COLORS[i % len(COLORS)]

        # Training loss
        steps, losses = get_metric_pairs(metrics, "step", "train/loss")
        if steps:
            ax1.plot(steps, losses, label=name, color=color, alpha=0.85, linewidth=1.5)

        # Validation loss
        steps, losses = get_metric_pairs(metrics, "step", "val/loss")
        if steps:
            ax2.plot(steps, losses, label=name, color=color, marker="o",
                     markersize=4, linewidth=1.5)
            has_val = True

    for ax, title in [(ax1, "Training Loss"), (ax2, "Validation Loss")]:
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    if not has_val:
        ax2.text(0.5, 0.5, "No validation data found",
                 transform=ax2.transAxes, ha="center", va="center", fontsize=12, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def build_summary(base_dir: Path) -> List[Dict]:
    """Build ranked summary for all experiments."""
    rows = []
    for exp_path in sorted(base_dir.iterdir()):
        if not exp_path.is_dir():
            continue
        metrics = load_metrics(str(exp_path))
        if not metrics:
            continue

        train_losses = [l for l in metrics.get("train/loss", []) if isinstance(l, float)]
        val_losses = [l for l in metrics.get("val/loss", []) if isinstance(l, float)]
        throughputs = [t for t in metrics.get("perf/tokens_per_sec", []) if isinstance(t, float)]
        steps = [s for s in metrics.get("step", []) if isinstance(s, float)]

        rows.append({
            "name": exp_path.name,
            "final_train_loss": train_losses[-1] if train_losses else float("inf"),
            "best_val_loss": min(val_losses) if val_losses else float("inf"),
            "best_val_ppl": math.exp(min(val_losses)) if val_losses else float("inf"),
            "avg_throughput": np.mean(throughputs) if throughputs else 0,
            "total_steps": int(max(steps)) if steps else 0,
        })
    return rows


def print_ranked_table(rows: List[Dict]):
    """Print a ranked markdown table sorted by best val loss."""
    rows = sorted(rows, key=lambda r: r["best_val_loss"])

    print(f"\n{'Rank':<5} {'Experiment':<25} {'Best Val Loss':<15} {'Val PPL':<12} "
          f"{'Final Train Loss':<18} {'Throughput (tok/s)':<20} {'Steps':<8}")
    print("-" * 105)

    for i, r in enumerate(rows, 1):
        val_loss = f"{r['best_val_loss']:.4f}" if r['best_val_loss'] < float("inf") else "N/A"
        val_ppl = f"{r['best_val_ppl']:.2f}" if r['best_val_ppl'] < float("inf") else "N/A"
        train_loss = f"{r['final_train_loss']:.4f}" if r['final_train_loss'] < float("inf") else "N/A"
        throughput = f"{r['avg_throughput']:,.0f}" if r['avg_throughput'] > 0 else "N/A"
        print(f"{i:<5} {r['name']:<25} {val_loss:<15} {val_ppl:<12} "
              f"{train_loss:<18} {throughput:<20} {r['total_steps']:<8}")


def print_suite_winners(rows: List[Dict]):
    """Print the winner of each ablation suite."""
    print(f"\n{'='*60}")
    print("ABLATION WINNERS (by best validation loss)")
    print(f"{'='*60}\n")

    for suite_name, exp_names in SUITES.items():
        suite_rows = [r for r in rows if r["name"] in exp_names]
        if not suite_rows:
            continue
        suite_rows.sort(key=lambda r: r["best_val_loss"])
        winner = suite_rows[0]
        worst = suite_rows[-1]

        print(f"  {suite_name}:")
        print(f"    Winner: {winner['name']:<25} (val loss: {winner['best_val_loss']:.4f})")
        if len(suite_rows) > 1:
            gap = worst["best_val_loss"] - winner["best_val_loss"]
            print(f"    Worst:  {worst['name']:<25} (val loss: {worst['best_val_loss']:.4f})")
            print(f"    Gap:    {gap:.4f}")
        print()


def save_markdown_report(rows: List[Dict], base_dir: Path):
    """Save a markdown report for the write-up."""
    rows_sorted = sorted(rows, key=lambda r: r["best_val_loss"])
    lines = ["# Ablation Study Results\n"]

    # Overall ranking
    lines.append("## Overall Ranking (by best validation loss)\n")
    lines.append("| Rank | Experiment | Best Val Loss | Val PPL | Final Train Loss | Throughput |")
    lines.append("|------|-----------|--------------|---------|-----------------|------------|")
    for i, r in enumerate(rows_sorted, 1):
        val = f"{r['best_val_loss']:.4f}" if r['best_val_loss'] < float("inf") else "N/A"
        ppl = f"{r['best_val_ppl']:.2f}" if r['best_val_ppl'] < float("inf") else "N/A"
        train = f"{r['final_train_loss']:.4f}" if r['final_train_loss'] < float("inf") else "N/A"
        tput = f"{r['avg_throughput']:,.0f}" if r['avg_throughput'] > 0 else "N/A"
        lines.append(f"| {i} | {r['name']} | {val} | {ppl} | {train} | {tput} |")

    # Per-suite winners
    lines.append("\n## Suite Winners\n")
    for suite_name, exp_names in SUITES.items():
        suite_rows = [r for r in rows_sorted if r["name"] in exp_names]
        if not suite_rows:
            continue
        winner = suite_rows[0]
        lines.append(f"- **{suite_name}**: `{winner['name']}` (val loss: {winner['best_val_loss']:.4f})")

    report_path = base_dir / "ablation_report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments")
    args = parser.parse_args()

    base = Path(args.exp_dir)
    if not base.exists():
        print(f"No experiments found at {base}")
        exit(1)

    # Build summary data
    rows = build_summary(base)
    if not rows:
        print("No experiment data found.")
        exit(1)

    # Print ranked table
    print("\n=== FULL RANKING ===")
    print_ranked_table(rows)

    # Print per-suite winners
    print_suite_winners(rows)

    # Generate per-suite plots
    print("=== Generating Per-Suite Plots ===\n")
    plots_dir = base / "plots"
    plots_dir.mkdir(exist_ok=True)

    for suite_name, exp_names in SUITES.items():
        available = [n for n in exp_names if (base / n).is_dir()]
        if available:
            safe_name = suite_name.lower().replace(" ", "_")
            plot_suite(suite_name, available, base, plots_dir / f"{safe_name}.png")

    # Save markdown report
    save_markdown_report(rows, base)
