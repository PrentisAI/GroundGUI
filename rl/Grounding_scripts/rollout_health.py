#!/usr/bin/env python
"""Rollout health for a plain OPSD run (no TtRT metrics in the log).

OPSD does not emit ttrt/bad_layout_rate, so the only per-step evidence of whether
the vLLM rollout server is serving a sane model is the logged student completions
themselves. A healthy Qwen3-VL-8B answers a grounding prompt with a tool call
containing exactly one "coordinate": [cx, cy] pair, i.e. exactly two runs of
digit characters in the argument. Anything else -- babble, CJK injection,
truncation, degenerate repetition -- is what a corrupted weight sync produces.

    bad = share of completions in the step with != 2 digit spans

Reading it: a healthy run stays close to 0 on every step; a corrupted weight
sync pushes the per-step rate far above the 0.10 "clean" threshold used below.

Usage:
    python Grounding_scripts/rollout_health.py output/gui-sd/v0-YYYYmmdd-HHMMSS
    python Grounding_scripts/rollout_health.py <dir> --tail 20
"""
import argparse
import json
import os
import re
import sys

DIGIT_SPAN = re.compile(r'\d+')


def coord_payload(text: str) -> str:
    """The part of the completion that should hold the coordinate.

    Slicing after "coordinate" avoids counting digits that belong to the tool
    schema or to the instruction echoed back by a degenerate completion.
    """
    i = text.find('"coordinate"')
    if i < 0:
        i = text.find('coordinate')
    return text if i < 0 else text[i:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', help='an output/<run>/vN-<ts> directory, or a completions.jsonl')
    ap.add_argument('--tail', type=int, default=0, help='only show the last N steps')
    args = ap.parse_args()

    path = args.run_dir
    if os.path.isdir(path):
        path = os.path.join(path, 'completions.jsonl')
    if not os.path.exists(path):
        print(f'no completions.jsonl at {path}', file=sys.stderr)
        return 1

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # the last line can be a partial write while training is live
                continue

    if not rows:
        print('completions.jsonl is empty (no optimizer step has been logged yet)')
        return 0

    per_step = []
    for r in rows:
        comps = r.get('completion') or []
        steps = r.get('step') or []
        step = steps[0] if steps else '?'
        if not comps:
            continue
        bad = sum(1 for c in comps if len(DIGIT_SPAN.findall(coord_payload(c))) != 2)
        per_step.append((step, bad / len(comps), len(comps)))

    shown = per_step[-args.tail:] if args.tail else per_step
    for step, bad, n in shown:
        flag = '' if bad <= 0.10 else ('  <-- SUSPECT' if bad <= 0.5 else '  <-- BAD')
        print(f'step {str(step):>4}  n={n:<4} bad={bad:.3f}{flag}')

    rates = [b for _, b, _ in per_step]
    clean = sum(1 for b in rates if b <= 0.10)
    print('-' * 46)
    print(f'steps logged      : {len(rates)}')
    print(f'mean bad rate     : {sum(rates) / len(rates):.4f}')
    print(f'clean steps (<=.1): {clean}/{len(rates)}  ({100 * clean / len(rates):.1f}%)')
    if len(rates) >= 20:
        last = rates[-20:]
        print(f'last 20 mean      : {sum(last) / len(last):.4f}')
    print('reference: a healthy run has mean bad rate near 0 and ~100% clean steps; a corrupted weight sync is far above 0.1')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
