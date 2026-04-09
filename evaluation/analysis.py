"""Analysis tools: per-suite loss curves, ranked ablation tables, best stack recommendation.

Uses training loss as primary metric (works even without validation data).
When val data is available, it's shown alongside for confirmation.

Usage:
    python -m evaluation.analysis --exp_dir ./experiments/
"""

import csv
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Map experiment name prefixes to ablation suites for grouped plotting
SUITES = {
    "Positional Encoding": ["pos_rope", "pos_alibi", "pos_learned"],
    "Attention Mechanism": ["attn_mha", "attn_gqa4", "attn_gqa2", "attn_mqa"],
    "Activation Function": ["act_swiglu", "act_gelu"],
    "Normalization": ["norm_pre_rmsnorm", "norm_pre_layernorm", "norm_post_rmsnorm", "norm_post_layernorm"],
    "Warmup Steps": ["warmup_100", "warmup_500", "warmup_1000", "warmup_2000"],
    "Learning Rate": ["lr_1e4", "lr_3e4", "lr_6e4", "lr_1e3"],
}

# Readable descriptions for the report
CHOICE_LABELS = {
    "pos_rope": "RoPE", "pos_alibi": "ALiBi", "pos_learned": "Learned",
    "attn_mha": "MHA (12 KV heads)", "attn_gqa4": "GQA-4", "attn_gqa2": "GQA-2", "attn_mqa": "MQA (1 KV head)",
    "act_swiglu": "SwiGLU", "act_gelu": "GELU",
    "norm_pre_rmsnorm": "Pre-norm + RMSNorm", "norm_pre_layernorm": "Pre-norm + LayerNorm",
    "norm_post_rmsnorm": "Post-norm + RMSNorm", "norm_post_layernorm": "Post-norm + LayerNorm",
    "warmup_100": "100 steps", "warmup_500": "500 steps", "warmup_1000": "1000 steps", "warmup_2000": "2000 steps",
    "lr_1e4": "1e-4", "lr_3e4": "3e-4", "lr_6e4": "6e-4", "lr_1e3": "1e-3",
}

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]


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
             if isinstance(x, (int, float)) and isinstance(y, (int, float))
             and x is not None and y is not None]
    if not pairs:
        return [], []
    return zip(*pairs)


def ranking_loss(row: Dict) -> float:
    """Best available loss for ranking: val if available, else train."""
    if row["best_val_loss"] < float("inf"):
        return row["best_val_loss"]
    return row["final_train_loss"]


def plot_suite(suite_name: str, exp_names: List[str], base_dir: Path, output_path: Path):
    """Plot training + validation loss for one ablation suite."""
    # Check if any experiment has val data
    has_val = False
    for name in exp_names:
        m = load_metrics(str(base_dir / name))
        if m and any(isinstance(v, float) for v in m.get("val/loss", [])):
            has_val = True
            break

    if has_val:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    fig.suptitle(f"Ablation: {suite_name}", fontsize=14, fontweight="bold")

    for i, name in enumerate(exp_names):
        exp_dir = str(base_dir / name)
        metrics = load_metrics(exp_dir)
        if not metrics:
            continue
        color = COLORS[i % len(COLORS)]
        label = CHOICE_LABELS.get(name, name)

        # Training loss
        steps, losses = get_metric_pairs(metrics, "step", "train/loss")
        if steps:
            ax1.plot(steps, losses, label=label, color=color, alpha=0.85, linewidth=1.5)

        # Validation loss (if panel exists)
        if has_val:
            steps, losses = get_metric_pairs(metrics, "step", "val/loss")
            if steps:
                ax2.plot(steps, losses, label=label, color=color, marker="o",
                         markersize=4, linewidth=1.5)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    if has_val:
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss")
        ax2.set_title("Validation Loss")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

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
            "final_train_ppl": math.exp(min(train_losses[-1], 20)) if train_losses else float("inf"),
            "avg_throughput": np.mean(throughputs) if throughputs else 0,
            "total_steps": int(max(steps)) if steps else 0,
        })
    return rows


def print_ranked_table(rows: List[Dict]):
    """Print ranked table, using best available loss."""
    has_val = any(r["best_val_loss"] < float("inf") for r in rows)
    rows = sorted(rows, key=ranking_loss)

    loss_col = "Best Val Loss" if has_val else "Final Train Loss"
    ppl_col = "Val PPL" if has_val else "Train PPL"

    print(f"\n{'Rank':<5} {'Experiment':<25} {loss_col:<18} {ppl_col:<12} "
          f"{'Throughput (tok/s)':<20} {'Steps':<8}")
    print("-" * 90)

    for i, r in enumerate(rows, 1):
        if has_val and r["best_val_loss"] < float("inf"):
            loss = f"{r['best_val_loss']:.4f}"
            ppl = f"{r['best_val_ppl']:.2f}"
        else:
            loss = f"{r['final_train_loss']:.4f}" if r['final_train_loss'] < float("inf") else "N/A"
            ppl = f"{r['final_train_ppl']:.2f}" if r['final_train_ppl'] < float("inf") else "N/A"
        throughput = f"{r['avg_throughput']:,.0f}" if r['avg_throughput'] > 0 else "N/A"
        print(f"{i:<5} {r['name']:<25} {loss:<18} {ppl:<12} "
              f"{throughput:<20} {r['total_steps']:<8}")


def get_suite_winner(rows: List[Dict], exp_names: List[str]) -> Dict:
    """Get the winning experiment from a suite."""
    suite_rows = [r for r in rows if r["name"] in exp_names]
    if not suite_rows:
        return None
    suite_rows.sort(key=ranking_loss)
    return suite_rows[0]


def print_suite_results(rows: List[Dict]):
    """Print detailed per-suite results with winner, ranking, and gaps."""
    has_val = any(r["best_val_loss"] < float("inf") for r in rows)
    metric_label = "val loss" if has_val else "train loss"

    print(f"\n{'='*60}")
    print(f"PER-SUITE RESULTS (ranked by {metric_label})")
    print(f"{'='*60}")

    for suite_name, exp_names in SUITES.items():
        suite_rows = [r for r in rows if r["name"] in exp_names]
        if not suite_rows:
            continue
        suite_rows.sort(key=ranking_loss)

        print(f"\n  {suite_name}:")
        for i, r in enumerate(suite_rows):
            loss = ranking_loss(r)
            label = CHOICE_LABELS.get(r["name"], r["name"])
            marker = " <-- BEST" if i == 0 else ""
            tput = f"{r['avg_throughput']:,.0f} tok/s" if r['avg_throughput'] > 0 else ""
            print(f"    {i+1}. {label:<30} loss: {loss:.4f}  {tput}{marker}")

        if len(suite_rows) > 1:
            gap = ranking_loss(suite_rows[-1]) - ranking_loss(suite_rows[0])
            print(f"    Gap (best→worst): {gap:.4f}")


def print_best_stack(rows: List[Dict]):
    """Combine winners from each suite into the recommended 'best stack'."""
    print(f"\n{'='*60}")
    print("RECOMMENDED BEST STACK")
    print("(combining the winner from each ablation suite)")
    print(f"{'='*60}\n")

    stack = {}
    for suite_name, exp_names in SUITES.items():
        winner = get_suite_winner(rows, exp_names)
        if winner:
            label = CHOICE_LABELS.get(winner["name"], winner["name"])
            loss = ranking_loss(winner)
            stack[suite_name] = (label, loss, winner["name"])
            print(f"  {suite_name + ':':<25} {label:<30} (loss: {loss:.4f})")

    # Print as a config snippet
    print(f"\n  --- As config overrides ---")
    config_map = {
        "Positional Encoding": ("model.pos_encoding", {"pos_rope": "rope", "pos_alibi": "alibi", "pos_learned": "learned"}),
        "Attention Mechanism": ("model.n_kv_heads", {"attn_mha": 12, "attn_gqa4": 4, "attn_gqa2": 2, "attn_mqa": 1}),
        "Activation Function": ("model.activation", {"act_swiglu": "swiglu", "act_gelu": "gelu"}),
        "Normalization": (None, {
            "norm_pre_rmsnorm": {"model.norm_position": "pre", "model.norm_type": "rmsnorm"},
            "norm_pre_layernorm": {"model.norm_position": "pre", "model.norm_type": "layernorm"},
            "norm_post_rmsnorm": {"model.norm_position": "post", "model.norm_type": "rmsnorm"},
            "norm_post_layernorm": {"model.norm_position": "post", "model.norm_type": "layernorm"},
        }),
        "Warmup Steps": ("training.warmup_steps", {"warmup_100": 100, "warmup_500": 500, "warmup_1000": 1000, "warmup_2000": 2000}),
        "Learning Rate": ("training.lr", {"lr_1e4": 1e-4, "lr_3e4": 3e-4, "lr_6e4": 6e-4, "lr_1e3": 1e-3}),
    }

    print("\n  python train.py --config configs/base.yaml \\")
    overrides = []
    for suite_name, (label, loss, name) in stack.items():
        key, mapping = config_map[suite_name]
        if key is None:
            # Normalization has multiple keys
            for k, v in mapping[name].items():
                overrides.append(f"    --override {k}={v}")
        else:
            overrides.append(f"    --override {key}={mapping[name]}")
    overrides.append(f"    --override name=best_stack")
    print(" \\\n".join(overrides))


def save_markdown_report(rows: List[Dict], base_dir: Path):
    """Save a markdown report for the write-up."""
    has_val = any(r["best_val_loss"] < float("inf") for r in rows)
    rows_sorted = sorted(rows, key=ranking_loss)

    lines = ["# Ablation Study Results\n"]
    metric_label = "Best Val Loss" if has_val else "Final Train Loss"
    ppl_label = "Val PPL" if has_val else "Train PPL"

    # Overall ranking
    lines.append(f"## Overall Ranking (by {metric_label.lower()})\n")
    lines.append(f"| Rank | Experiment | {metric_label} | {ppl_label} | Throughput |")
    lines.append("|------|-----------|--------------|---------|------------|")
    for i, r in enumerate(rows_sorted, 1):
        loss = ranking_loss(r)
        loss_s = f"{loss:.4f}" if loss < float("inf") else "N/A"
        ppl = math.exp(min(loss, 20)) if loss < float("inf") else "N/A"
        ppl_s = f"{ppl:.2f}" if isinstance(ppl, float) else "N/A"
        tput = f"{r['avg_throughput']:,.0f}" if r['avg_throughput'] > 0 else "N/A"
        lines.append(f"| {i} | {r['name']} | {loss_s} | {ppl_s} | {tput} |")

    # Per-suite winners
    lines.append("\n## Suite Winners\n")
    for suite_name, exp_names in SUITES.items():
        winner = get_suite_winner(rows, exp_names)
        if winner:
            label = CHOICE_LABELS.get(winner["name"], winner["name"])
            loss = ranking_loss(winner)
            lines.append(f"- **{suite_name}**: `{label}` (loss: {loss:.4f})")

    # Best stack
    lines.append("\n## Recommended Best Stack\n")
    lines.append("Combining the winner from each independent ablation:\n")
    for suite_name, exp_names in SUITES.items():
        winner = get_suite_winner(rows, exp_names)
        if winner:
            label = CHOICE_LABELS.get(winner["name"], winner["name"])
            lines.append(f"- {suite_name}: **{label}**")

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

    has_val = any(r["best_val_loss"] < float("inf") for r in rows)
    metric_used = "validation loss" if has_val else "training loss (no val data available)"
    print(f"\nRanking metric: {metric_used}")

    # Print ranked table
    print("\n=== FULL RANKING ===")
    print_ranked_table(rows)

    # Print per-suite results
    print_suite_results(rows)

    # Print best stack
    print_best_stack(rows)

    # Generate per-suite plots (only training loss panel if no val data)
    print(f"\n=== Generating Per-Suite Plots ===\n")
    plots_dir = base / "plots"
    plots_dir.mkdir(exist_ok=True)

    for suite_name, exp_names in SUITES.items():
        available = [n for n in exp_names if (base / n).is_dir()]
        if available:
            safe_name = suite_name.lower().replace(" ", "_")
            plot_suite(suite_name, available, base, plots_dir / f"{safe_name}.png")

    # Save markdown report
    save_markdown_report(rows, base)
