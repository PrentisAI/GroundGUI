#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-score an existing result file under the other GROUNDING_OFFICIAL_SCORING mode,
without re-running inference.

Usage:
    python rescore.py --result output/<model>/<benchmark>/<result>.json
    python rescore.py --result <file> --max-pixels 16777216 --factor 32

The input must be a result file written with GROUNDING_OFFICIAL_SCORING=1. The scoring
mode is not stored in result files, so this script is also the way to recover the other
number for an existing run.

Scoring is pure post-processing of the model output. A result file stores the raw
`response` and `gt_bbox` in resized pixel space; the only missing piece,
`processed_imgsize`, is recomputed deterministically with the same smart_resize from the
original image size. The result is therefore not an approximation: it is identical, sample
by sample, to re-running inference and scoring again, at no GPU cost.

The two modes (see GUIOwlPrompt.calculate_metrics):
  =1  official: the GT box is scaled to 0-1000 space and rounded OUTWARD (floor on the
      lower edges, ceil on the upper edges); the predicted point keeps its raw 0-1000
      value. Outward rounding loosens each edge by at most one unit.
  =0  exact: the predicted point is scaled back to pixel space and tested for containment
      in the unrounded float GT box.
=0 is strictly tighter, so its score is always <= the =1 score.

MAX_IMAGE_PIXELS and IMAGE_FACTOR are not stored in result files either. Unless given via
--max-pixels / --factor, they are searched over a list of common values, picking the pair
that reproduces the stored `correct` field for every sample under =1.

Self-check: if no pair reproduces the stored =1 scores exactly, the recomputation chain
(image sizes / parsing / coordinate mapping) does not match the original run, the =0
number cannot be trusted, and the script exits with status 1.
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_vl_utils import smart_resize
from data import load_benchmark_info
from prompts import get_prompt_processor, is_point_inside_element, _official_gt_1000
from imgdims_helper import dims   # noqa


def score(gt_bbox, point, proc, official):
    if point is None:
        return 0
    if point[0] >= 0 and point[1] >= 0 and proc:
        if official:
            g = _official_gt_1000(gt_bbox, proc)
            if g is not None:
                gt_bbox = g                       # point keeps its raw 0-1000 value
            else:
                point = [point[0] / 1000 * proc[0], point[1] / 1000 * proc[1]]
        else:
            point = [point[0] / 1000 * proc[0], point[1] / 1000 * proc[1]]
    return int(is_point_inside_element(gt_bbox, point))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result', required=True)
    p.add_argument('--benchmark', default=None, help='default: the benchmark stored in the result file')
    p.add_argument('--prompt', default=None, help='default: the prompt stored in the result file')
    p.add_argument('--max-pixels', type=int, default=None, help='MAX_IMAGE_PIXELS of the original run; inferred automatically if omitted')
    p.add_argument('--factor', type=int, default=None, help='IMAGE_FACTOR of the original run; inferred automatically if omitted')
    a = p.parse_args()

    d = json.load(open(a.result))
    bench = a.benchmark or d['benchmark']
    prompt = a.prompt or d['prompt']
    proc_cls = get_prompt_processor(prompt)
    info = load_benchmark_info(bench)
    root = info['image_root']

    det = d['detailed_results']
    det = list(det.values()) if isinstance(det, dict) else det

    n = len(det)
    old_correct = sum(int(r.get('correct', 0)) for r in det)

    # MAX_IMAGE_PIXELS / IMAGE_FACTOR are not stored in the result file, but they determine
    # the resized size, and a wrong guess skews every coordinate mapping. Search known values
    # for the pair that reproduces =1 for every sample. This doubles as the self-check: if
    # none does, the recomputation chain is wrong and no =0 number is reported.
    CAND_PX = [a.max_pixels] if a.max_pixels else [
        6553600, 4233600, 12845056, 10035200, 2109744, 16777216, 1003520]
    CAND_F = [a.factor] if a.factor else [28, 32]
    sizes = {}
    for r in det:
        wh = dims(os.path.join(root, r['image']))
        if wh is None:
            print(f"missing image: {r['image']}", file=sys.stderr); sys.exit(2)
        sizes[r['image']] = wh
    pts = [proc_cls.extract_coordinates(r['response']) for r in det]

    best = None
    for mp in CAND_PX:
        for fa in CAND_F:
            mm = 0
            for r, pt in zip(det, pts):
                w, h = sizes[r['image']]
                nh, nw = smart_resize(h, w, max_pixels=mp, factor=fa)
                if score(r['gt_bbox'], pt, (nw, nh), True) != int(r.get('correct', 0)):
                    mm += 1
            if best is None or mm < best[0]:
                best = (mm, mp, fa)
            if mm == 0:
                break
        if best[0] == 0:
            break
    mismatch, MP, FA = best
    chk1 = c0 = 0
    for r, pt in zip(det, pts):
        w, h = sizes[r['image']]
        nh, nw = smart_resize(h, w, max_pixels=MP, factor=FA)
        chk1 += score(r['gt_bbox'], pt, (nw, nh), True)
        c0 += score(r['gt_bbox'], pt, (nw, nh), False)

    print(f'{bench:<16} prompt={prompt}  [inferred max_pixels={MP} factor={FA}]')
    print(f'  stored     (OFFICIAL=1) : {old_correct}/{n} = {old_correct/n*100:.2f}%')
    print(f'  recomputed (OFFICIAL=1) : {chk1}/{n} = {chk1/n*100:.2f}%   per-sample mismatches {mismatch}')
    if mismatch:
        print('  self-check FAILED: recomputation does not match the original run; the OFFICIAL=0 number cannot be trusted', file=sys.stderr)
        sys.exit(1)
    print(f'  recomputed (OFFICIAL=0) : {c0}/{n} = {c0/n*100:.2f}%   change vs =1 {(c0-chk1)/n*100:+.2f} pp')


if __name__ == '__main__':
    main()
