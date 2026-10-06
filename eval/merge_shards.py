#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stitch the per-shard result files written by `eval.py` under
EVAL_NUM_SHARDS/EVAL_SHARD_ID back into one ordinary result file.

Each shard already keys detailed_results by the ORIGINAL global sample index, so
merging is a dict union plus a recomputation of the hierarchical statistics over
the full dataset -- the merged file is then indistinguishable from an unsharded
run and every downstream consumer (e.g. rescore.py) works on it unchanged.

  python merge_shards.py --model <model_name> \
                         --benchmark ui-vision [--num-shards 6] [--newer-than EPOCH]
"""
import argparse
import json
import os
import re
from datetime import datetime

from data import (load_benchmark_info, load_dataset, standardize_sample,
                  calculate_hierarchical_statistics, reorder_stats_for_output)
from prompts import get_prompt_processor

SHARD_RE = re.compile(r"_shard(\d+)of(\d+)_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="output/<model>/ directory name")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--num-shards", type=int, default=0, help="0 = infer from filenames")
    ap.add_argument("--newer-than", type=float, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    d = args.output_dir or os.path.join("output", args.model, args.benchmark)
    shards = {}
    for f in sorted(os.listdir(d)):
        m = SHARD_RE.search(f)
        if not m or not f.endswith(".json"):
            continue
        p = os.path.join(d, f)
        if args.newer_than is not None and os.path.getmtime(p) < args.newer_than:
            continue
        sid, tot = int(m.group(1)), int(m.group(2))
        if args.num_shards and tot != args.num_shards:
            continue
        # keep the NEWEST file per shard id
        if sid not in shards or os.path.getmtime(p) > os.path.getmtime(shards[sid]):
            shards[sid] = p
    if not shards:
        raise SystemExit(f"no shard files under {d}")

    total = {int(SHARD_RE.search(os.path.basename(p)).group(2)) for p in shards.values()}
    assert len(total) == 1, f"mixed shard counts: {total}"
    n_shards = total.pop()
    missing = sorted(set(range(n_shards)) - set(shards))
    if missing:
        raise SystemExit(f"MISSING shards {missing} of {n_shards} -- refusing to merge a partial run")

    merged, first = {}, None
    for sid in sorted(shards):
        with open(shards[sid], encoding="utf-8") as f:
            j = json.load(f)
        first = first or j
        assert j["prompt"] == first["prompt"] and j["model_path"] == first["model_path"], \
            f"shard {sid} disagrees on prompt/model with shard {min(shards)}"
        dup = set(j["detailed_results"]) & set(merged)
        assert not dup, f"shard {sid} overlaps earlier shards on {sorted(dup)[:5]}"
        merged.update(j["detailed_results"])
        print(f"  shard {sid}/{n_shards}: {len(j['detailed_results']):5d} rows  <- {os.path.basename(shards[sid])}")

    # rebuild statistics over the FULL dataset
    info = load_benchmark_info(args.benchmark)
    dataset = [standardize_sample(s, args.benchmark) for s in load_dataset(info)]
    if len(merged) != len(dataset):
        raise SystemExit(f"merged {len(merged)} rows but dataset has {len(dataset)} -- shards do not cover it")
    pp = get_prompt_processor(first["prompt"])
    results = {int(k): v for k, v in merged.items()}
    stats = reorder_stats_for_output(calculate_hierarchical_statistics(results, dataset, pp))

    out = dict(first)
    out.pop("shard", None)
    out["statistics"] = stats
    out["detailed_results"] = {str(k): results[k] for k in sorted(results)}
    out["merged_from"] = [os.path.basename(shards[s]) for s in sorted(shards)]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    t = first["args"].get("temperature") or 0
    name = f"{ts}{'_t'+str(t).replace('.', '-') if t else ''}_{first['prompt']}.json"
    op = os.path.join(d, name)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    acc = {k: v for k, v in stats.items() if k.endswith("_accuracy")}
    print(f"merged {len(results)} rows from {n_shards} shards -> {op}")
    print(f"Overall accuracy: {acc}")


if __name__ == "__main__":
    main()
