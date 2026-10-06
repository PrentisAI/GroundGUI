"""
Evaluate a training run: for the latest version directory of ./output/<run_name>, score every
checkpoint's infer_result/<dataset>.jsonl on each requested benchmark and print a summary table.

Usage:
    python swift/metrics/total_metric.py --run_name my_grpo_run
    python swift/metrics/total_metric.py --run_name my_opsd_run --metric navi_qwen3
    python swift/metrics/total_metric.py --run_name my_grpo_run --datasets screenspotpro uivision
"""

import os
import sys
import json
import argparse
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from swift.utils import read_from_jsonl
from swift.metrics.screenspotpro_metric import compute_screenspotpro
from swift.metrics.screenspotv2_metric import compute_screenspotv2
from swift.metrics.uivision_metric import compute_uivision
from swift.metrics.osworldg_metric import compute_osworldg
from swift.metrics.mmbench_metric import compute_mmbench


# dataset name -> (result file name, compute function, metric prefix)
DATASET_REGISTRY = {
    "screenspotpro": ("screenspotpro.jsonl", compute_screenspotpro, "screenspotpro"),
    "screenspotv2":  ("screenspotv2.jsonl",  compute_screenspotv2,  "screenspotv2"),
    "uivision":      ("uivision.jsonl",      compute_uivision,      "uivision"),
    "osworldg":      ("osworldg.jsonl",       compute_osworldg,      "osworldg"),
    "osworldg_r":    ("osworldg_r.jsonl",     compute_osworldg,      "osworldg"),
    "mmbench":       ("mmbench.jsonl",        compute_mmbench,       "mmbench"),
}

OUTPUT_BASE = "./output"


def get_latest_vdir(run_name):
    base_dir = os.path.join(OUTPUT_BASE, run_name)
    if not os.path.exists(base_dir):
        print(f"ERROR: run_name '{run_name}' not found in {OUTPUT_BASE}")
        return None
    v_dirs = sorted(glob(os.path.join(base_dir, "v*")), key=os.path.getmtime, reverse=True)
    return v_dirs[0] if v_dirs else None


def get_checkpoints(vdir):
    ckpts = glob(os.path.join(vdir, "checkpoint-*"))
    ckpts = sorted(ckpts, key=lambda x: int(os.path.basename(x).split("-")[1]))
    return ckpts


def evaluate_checkpoint(ckpt_path, datasets, metric_suffix):
    """Evaluate one checkpoint on the given datasets and return {dataset: overall metrics or None}."""
    ckpt_name = os.path.basename(ckpt_path)
    results = {}

    for ds_name in datasets:
        if ds_name not in DATASET_REGISTRY:
            print(f"  WARNING: unknown dataset '{ds_name}', skipping.")
            continue

        filename, compute_fn, prefix = DATASET_REGISTRY[ds_name]
        jsonl_path = os.path.join(ckpt_path, "infer_result", filename)

        if not os.path.exists(jsonl_path):
            print(f"  [{ds_name}] SKIP - file not found: {jsonl_path}")
            results[ds_name] = None
            continue

        metric = f"{prefix}_{metric_suffix}"
        print(f"\n{'='*60}")
        print(f"  [{ckpt_name}] Evaluating {ds_name} (metric={metric})")
        print(f"{'='*60}")

        data_list = read_from_jsonl(jsonl_path)
        overall = compute_fn(data_list, metric)
        results[ds_name] = overall

    return results


def print_summary_table(all_results, datasets):
    """Print a checkpoint x dataset summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # Header
    header = f"{'Checkpoint':<25}"
    for ds in datasets:
        header += f"  {ds:>15}"
    print(header)
    print("-" * len(header))

    # One row per checkpoint
    for ckpt_name, results in all_results.items():
        row = f"{ckpt_name:<25}"
        for ds in datasets:
            if ds not in results or results[ds] is None:
                row += f"  {'N/A':>15}"
            else:
                overall = results[ds]
                # Use Total Acc, or Average Acc for MMBench
                acc = overall.get("Total Acc", overall.get("Average Acc", "N/A"))
                row += f"  {acc + '%':>15}"
        print(row)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate every checkpoint of a run on the grounding benchmarks")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Run name: a directory under ./output (e.g. my_grpo_run)")
    parser.add_argument("--metric", type=str, default="navi_qwen3",
                        help="Metric suffix: navi_qwen3, ground_qwen3 or venus (default: navi_qwen3)")
    parser.add_argument("--datasets", nargs="+", type=str,
                        default=["screenspotpro", "screenspotv2", "uivision", "osworldg", "osworldg_r", "mmbench"],
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Datasets to evaluate (default: all six)")
    args = parser.parse_args()

    # Latest version directory
    vdir = get_latest_vdir(args.run_name)
    if not vdir:
        print(f"ERROR: no vdir found for '{args.run_name}'")
        return
    print(f"Run: {args.run_name}")
    print(f"Vdir: {vdir}")
    print(f"Datasets: {args.datasets}")
    print(f"Metric suffix: {args.metric}")

    # All checkpoints
    ckpts = get_checkpoints(vdir)
    if not ckpts:
        print(f"ERROR: no checkpoints found in {vdir}")
        return
    print(f"Found {len(ckpts)} checkpoint(s): {[os.path.basename(c) for c in ckpts]}")

    # Evaluate each checkpoint
    all_results = {}
    for ckpt in ckpts:
        ckpt_name = os.path.basename(ckpt)
        results = evaluate_checkpoint(ckpt, args.datasets, args.metric)
        all_results[ckpt_name] = results

    # Summary table
    print_summary_table(all_results, args.datasets)


if __name__ == "__main__":
    main()
