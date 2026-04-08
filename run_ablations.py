"""Run systematic ablation studies.

Each ablation changes exactly ONE variable from the base config,
enabling clean causal attribution of any performance differences.

Usage:
    python run_ablations.py --suite all
    python run_ablations.py --suite positional_encoding
    python run_ablations.py --suite attention
"""

import argparse
import subprocess
import sys
from typing import List, Dict


# Each suite defines a set of experiments as (name, overrides) pairs.
# Every experiment shares the base config except for the listed overrides.
ABLATION_SUITES = {
    "positional_encoding": [
        ("pos_rope",    {"model.pos_encoding": "rope"}),
        ("pos_alibi",   {"model.pos_encoding": "alibi"}),
        ("pos_learned", {"model.pos_encoding": "learned"}),
    ],
    "attention": [
        ("attn_mha",      {"model.n_kv_heads": 12}),   # MHA: all heads unique
        ("attn_gqa4",     {"model.n_kv_heads": 4}),    # GQA: 4 KV groups
        ("attn_gqa2",     {"model.n_kv_heads": 2}),    # GQA: 2 KV groups
        ("attn_mqa",      {"model.n_kv_heads": 1}),    # MQA: single KV head
    ],
    "activation": [
        ("act_swiglu", {"model.activation": "swiglu"}),
        ("act_gelu",   {"model.activation": "gelu"}),
    ],
    "normalization": [
        ("norm_pre_rmsnorm",   {"model.norm_position": "pre",  "model.norm_type": "rmsnorm"}),
        ("norm_pre_layernorm", {"model.norm_position": "pre",  "model.norm_type": "layernorm"}),
        ("norm_post_rmsnorm",  {"model.norm_position": "post", "model.norm_type": "rmsnorm"}),
        ("norm_post_layernorm",{"model.norm_position": "post", "model.norm_type": "layernorm"}),
    ],
    "warmup": [
        ("warmup_100",  {"training.warmup_steps": 100}),
        ("warmup_500",  {"training.warmup_steps": 500}),
        ("warmup_1000", {"training.warmup_steps": 1000}),
        ("warmup_2000", {"training.warmup_steps": 2000}),
    ],
    "lr": [
        ("lr_1e4",  {"training.lr": 1e-4}),
        ("lr_3e4",  {"training.lr": 3e-4}),
        ("lr_6e4",  {"training.lr": 6e-4}),
        ("lr_1e3",  {"training.lr": 1e-3}),
    ],
}

# "all" runs every suite
ABLATION_SUITES["all"] = []
for suite_name, experiments in list(ABLATION_SUITES.items()):
    if suite_name != "all":
        ABLATION_SUITES["all"].extend(experiments)


def run_experiment(name: str, overrides: Dict[str, any], base_config: str):
    """Launch a single training run as a subprocess."""
    cmd = [sys.executable, "train.py", "--config", base_config]
    cmd.extend(["--override", f"name={name}"])
    for key, val in overrides.items():
        cmd.extend(["--override", f"{key}={val}"])

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Overrides: {overrides}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: {name} exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--suite", type=str, required=True,
                        choices=list(ABLATION_SUITES.keys()),
                        help="Which ablation suite to run")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max_steps for all experiments (e.g. 10000 for quick screening)")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    experiments = ABLATION_SUITES[args.suite]
    print(f"Ablation suite: {args.suite} ({len(experiments)} experiments)")

    for name, overrides in experiments:
        if args.max_steps is not None:
            overrides = {**overrides, "training.max_steps": args.max_steps}
        if args.dry_run:
            print(f"  [DRY RUN] {name}: {overrides}")
        else:
            run_experiment(name, overrides, args.base_config)

    print(f"\nAll experiments complete. Run analysis:")
    print(f"  python -m evaluation.analysis --exp_dir ./experiments/")


if __name__ == "__main__":
    main()
