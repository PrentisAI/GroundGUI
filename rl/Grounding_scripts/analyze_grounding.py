#!/usr/bin/env python3
"""Turn swift infer result jsonl into per-sample grounding records + a summary.

Stage 1 of the error analysis. Reads exp/<run>/infer/<ds>.jsonl (the output of
Grounding_scripts/test_6gpu.sh) and writes, per dataset:

    exp/<run>/analysis/<ds>.records.jsonl   one row per sample, all derived fields
    exp/<run>/analysis/summary.json         accuracy tables + error-type histograms

Correctness is replicated from swift/metrics/*_metric.py exactly. The three
benchmarks disagree with each other on scoring convention and getting this wrong
silently mislabels every boundary case:

    screenspotv2 / screenspotpro   pixel space, int() truncation, STRICT  <
    osworldg / osworldg_r          pixel space, int() truncation, inclusive <=
    mmbench                        NORMALISED space, no truncation,   inclusive <=

Run with --verify to assert our accuracy against the shipped metric functions.

Error taxonomy (ordered, first match wins, mutually exclusive and exhaustive):

    A1_NO_TOOL_CALL     no parsable coordinate in the response
    A2_REFUSAL_CLICKED  the item requires declining; the model clicked
    B1_METRIC_CEILING   the model was right and the metric said no
                          .strict_inequality_boundary  point on the box edge
                          .off_by_one_code             +-1 on the 0-1000 grid would hit
                          .prose_abstain_on_refusal    declined in prose, unparsable
    C1_PRECISION_MISS   missed by less than the target's own size on both axes
    C2_ROW_MISS         exactly inside the target's row band, horizontally away
    C3_COL_MISS         exactly inside the target's column band, vertically away
    C4_REGION_MISS      diagonal miss, still closer than the screen centre is
    C5_ELSEWHERE        no closer to the target than clicking the middle of the screen

Band membership is exact (excess == 0), not a tolerance: exact row/column
alignment is the informative signal, while a positive tolerance mostly admits
misses that line up with the target only by chance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from swift.custom_utils.format_func import extract_action  # noqa: E402

STRICT = {'screenspotv2', 'screenspotpro'}   # metric uses '<', not '<='
NORMSPACE = {'mmbench'}                      # metric scores in [0,1], no int()
OSWORLDG = {'osworldg', 'osworldg_r'}
ALL_DATASETS = ['screenspotv2', 'screenspotpro', 'mmbench', 'osworldg', 'osworldg_r']

CATEGORIES = ['A1_NO_TOOL_CALL', 'A2_REFUSAL_CLICKED', 'B1_METRIC_CEILING',
              'C1_PRECISION_MISS', 'C2_ROW_MISS', 'C3_COL_MISS',
              'C4_REGION_MISS', 'C5_ELSEWHERE']

# Vision-token geometry. IMAGE_MAX_TOKEN_NUM is a *merged token* budget; swift
# converts it to pixels with factor = patch_size * spatial_merge_size = 16*2
# (swift/model/models/qwen.py:705-716). One merged token is a 32x32 pixel cell
# of the *resized* image.
IMAGE_FACTOR = 32
MAX_PIXELS = 10000 * IMAGE_FACTOR ** 2       # IMAGE_MAX_TOKEN_NUM=10000 in test_6gpu.sh
MIN_PIXELS = 4 * IMAGE_FACTOR ** 2


def smart_resize(w: int, h: int, factor=IMAGE_FACTOR,
                 min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """qwen_vl_utils.smart_resize -> (w_bar, h_bar). Returns the resized size."""
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return w_bar, h_bar


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def pred_pixel(coord, size):
    w, h = size
    return [int(coord[0] / 1000 * w), int(coord[1] / 1000 * h)]


def point_in_polygon(p, poly):
    """Ray casting, identical to osworldg_metric._is_point_in_polygon."""
    x, y = p
    n = len(poly) // 2
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i * 2], poly[i * 2 + 1]
        xj, yj = poly[j * 2], poly[j * 2 + 1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def polygon_bbox(poly):
    xs, ys = poly[0::2], poly[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def dist_point_segment(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def dist_to_polygon(p, poly):
    if point_in_polygon(p, poly):
        return 0.0
    n = len(poly) // 2
    pts = [(poly[i * 2], poly[i * 2 + 1]) for i in range(n)]
    return min(dist_point_segment(p, pts[i], pts[(i + 1) % n]) for i in range(n))


def region_distance(p, bounds, polygon):
    if polygon is not None:
        return dist_to_polygon(p, polygon)
    dx = max(bounds[0] - p[0], 0, p[0] - bounds[2])
    dy = max(bounds[1] - p[1], 0, p[1] - bounds[3])
    return math.hypot(dx, dy)


def hit(ds, box_type, norm_xy, bounds, polygon, size):
    """Reproduce the shipped metric's decision exactly."""
    x, y = norm_xy
    w, h = size
    if box_type == 'refusal':
        return x < 0 and y < 0
    if box_type == 'polygon':
        return point_in_polygon([int(x / 1000 * w), int(y / 1000 * h)], polygon)
    if ds in NORMSPACE:
        return (bounds[0] / w <= x / 1000 <= bounds[2] / w
                and bounds[1] / h <= y / 1000 <= bounds[3] / h)
    px, py = int(x / 1000 * w), int(y / 1000 * h)
    if ds in STRICT:
        return bounds[0] < px < bounds[2] and bounds[1] < py < bounds[3]
    return bounds[0] <= px <= bounds[2] and bounds[1] <= py <= bounds[3]


# --------------------------------------------------------------------------
def build_records(ds: str, rows: list) -> list:
    recs = []
    for i, row in enumerate(rows):
        para = row['additional_paras']
        para = json.loads(para) if isinstance(para, str) else para
        size = para['image_size']
        W, H = size
        gt = row['solution']['arguments']['coordinate']

        images = row['images']
        img = images[0] if isinstance(images, list) else images
        img_path = img['path'] if isinstance(img, dict) else img

        box_type = para.get('box_type', 'bbox')
        polygon = None
        if box_type == 'polygon':
            polygon = list(gt)
            bounds = polygon_bbox(polygon)
        elif box_type == 'refusal':
            bounds = [0.0, 0.0, 0.0, 0.0]
        else:
            bounds = list(gt)

        action = extract_action(row['response'])
        parse_ok = action != 'no action'
        norm = px = None
        if parse_ok:
            try:
                c = action['arguments']['coordinate']
                norm = [float(c[0]), float(c[1])]
                px = pred_pixel(norm, size)
            except Exception:
                parse_ok = False

        correct = bool(parse_ok and hit(ds, box_type, norm, bounds, polygon, size))

        rw, rh = smart_resize(W, H)
        scale = rw / W

        rec = {
            'dataset': ds, 'idx': i, 'id': para.get('id', i),
            'image': img_path, 'W': W, 'H': H,
            'instruction': row['messages'][1]['content'],
            'response': row['response'],
            'box_type': box_type, 'gt': gt, 'bounds': bounds, 'polygon': polygon,
            'parse_ok': parse_ok, 'norm': norm, 'pred_px': px,
            'correct': correct, 'scale': scale,
            'paras': para,
        }

        if box_type == 'refusal':
            rec.update({k: None for k in
                        ('ex', 'ey', 'nx', 'ny', 'u', 'axis', 'd', 'd_tok',
                         'd_ctr', 'rho', 'tok_area', 'sub_token', 'off_x', 'off_y')})
        else:
            bw, bh = bounds[2] - bounds[0], bounds[3] - bounds[1]
            cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
            # Half-extents floored at half a decoder step, so a target thinner
            # than the 0-1000 grid does not produce an infinite nx.
            hw = max(bw / 2, W / 2000)
            hh = max(bh / 2, H / 2000)
            tok_area = bw * bh * scale ** 2 / IMAGE_FACTOR ** 2
            rec['tok_area'] = tok_area
            rec['sub_token'] = tok_area < 1
            d_ctr = region_distance([W / 2, H / 2], bounds, polygon)
            rec['d_ctr'] = d_ctr
            if parse_ok:
                ex = max(bounds[0] - px[0], 0, px[0] - bounds[2])
                ey = max(bounds[1] - px[1], 0, px[1] - bounds[3])
                d = region_distance(px, bounds, polygon)
                nx, ny = ex / hw, ey / hh
                rec.update({
                    'ex': ex, 'ey': ey, 'nx': nx, 'ny': ny,
                    'd': d, 'd_tok': d * scale / IMAGE_FACTOR,
                    'rho': (d / d_ctr) if d_ctr > 0 else float('inf'),
                    'u': nx if ey == 0 else (ny if ex == 0 else max(nx, ny)),
                    'axis': ('both' if ex == 0 and ey == 0 else
                             'row' if ey == 0 else 'col' if ex == 0 else 'off'),
                    'off_x': (px[0] - cx) / hw,
                    'off_y': (px[1] - cy) / hh,
                })
            else:
                rec.update({k: None for k in
                            ('ex', 'ey', 'nx', 'ny', 'u', 'axis', 'd', 'd_tok',
                             'rho', 'off_x', 'off_y')})
        recs.append(rec)
    return recs


def classify(r):
    """(category, subtype) for a failure; (None, None) if correct."""
    if r['correct']:
        return None, None

    if not r['parse_ok']:
        # Declining in prose on an item that *asks* for a refusal is correct
        # behaviour that the metric cannot express -- it demands negative
        # coordinates. That is a metric ceiling, not a model failure.
        if r['box_type'] == 'refusal':
            return 'B1_METRIC_CEILING', 'prose_abstain_on_refusal'
        return 'A1_NO_TOOL_CALL', None

    if r['box_type'] == 'refusal':
        return 'A2_REFUSAL_CLICKED', None

    ds, b, px = r['dataset'], r['bounds'], r['pred_px']
    if ds in STRICT and b[0] <= px[0] <= b[2] and b[1] <= px[1] <= b[3]:
        return 'B1_METRIC_CEILING', 'strict_inequality_boundary'

    for dk in (-1, 0, 1):
        for dm in (-1, 0, 1):
            if dk == dm == 0:
                continue
            if hit(ds, r['box_type'], [r['norm'][0] + dk, r['norm'][1] + dm],
                   b, r['polygon'], [r['W'], r['H']]):
                return 'B1_METRIC_CEILING', 'off_by_one_code'

    if max(r['nx'], r['ny']) <= 1.0:
        return 'C1_PRECISION_MISS', None
    if r['ey'] == 0:
        return 'C2_ROW_MISS', None
    if r['ex'] == 0:
        return 'C3_COL_MISS', None
    return ('C4_REGION_MISS', None) if r['rho'] < 1.0 else ('C5_ELSEWHERE', None)


# --------------------------------------------------------------------------
def verify_against_metrics(ds: str, rows: list, recs: list) -> str:
    import contextlib, io
    from swift.metrics.screenspotv2_metric import compute_screenspotv2
    from swift.metrics.screenspotpro_metric import compute_screenspotpro
    from swift.metrics.mmbench_metric import compute_mmbench
    from swift.metrics.osworldg_metric import compute_osworldg
    fn, prefix = {'screenspotv2': (compute_screenspotv2, 'screenspotv2'),
                  'screenspotpro': (compute_screenspotpro, 'screenspotpro'),
                  'mmbench': (compute_mmbench, 'mmbench'),
                  'osworldg': (compute_osworldg, 'osworldg'),
                  'osworldg_r': (compute_osworldg, 'osworldg')}[ds]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        official = fn(rows, f'{prefix}_navi_qwen3')
    key = 'Total Acc' if 'Total Acc' in official else 'Average Acc'
    if key == 'Average Acc':
        groups = defaultdict(list)
        for r in recs:
            groups[r['paras'].get('grounding_type', 'basic')].append(r['correct'])
        ours = sum(sum(v) / len(v) for v in groups.values()) / len(groups) * 100
    else:
        ours = sum(r['correct'] for r in recs) / len(recs) * 100
    delta = abs(ours - float(official[key]))
    return (f'  verify {ds}: ours={ours:.2f} official {key}={official[key]} '
            f'[{"OK" if delta < 0.01 else "MISMATCH"}]')


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--exp-dir', required=True)
    p.add_argument('-d', '--datasets', nargs='+', default=ALL_DATASETS)
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()

    exp = os.path.abspath(args.exp_dir)
    infer_dir = os.path.join(exp, 'infer')
    out_dir = os.path.join(exp, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    expected = {'screenspotv2': 1272, 'screenspotpro': 1581, 'mmbench': 3594,
                'osworldg': 564, 'osworldg_r': 564}
    summary = {'datasets': {}}

    for ds in args.datasets:
        path = os.path.join(infer_dir, f'{ds}.jsonl')
        if not os.path.exists(path):
            print(f'{ds}: SKIP (no {path})')
            continue
        rows = [json.loads(l) for l in open(path)]
        # The input files are sorted by facet (mmbench is all `basic` then all
        # `advanced`), so a prefix is never a representative sample. Refuse to
        # report a partial file rather than print a misleading number.
        if len(rows) != expected[ds]:
            print(f'{ds}: REFUSING partial file ({len(rows)}/{expected[ds]} rows). '
                  'Inputs are facet-sorted; a prefix is not a random sample.')
            continue

        recs = build_records(ds, rows)
        for r in recs:
            r['error_type'], r['error_sub'] = classify(r)

        nf = sum(not r['correct'] for r in recs)
        counts = Counter(r['error_type'] for r in recs if not r['correct'])
        assert sum(counts.values()) == nf, f'{ds}: taxonomy not exhaustive'

        with open(os.path.join(out_dir, f'{ds}.records.jsonl'), 'w') as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        n = len(recs)
        ncorr = n - nf
        subs = Counter(r['error_sub'] for r in recs
                       if r['error_type'] == 'B1_METRIC_CEILING')
        entry = {
            'n': n, 'correct': ncorr, 'accuracy': round(ncorr / n * 100, 2),
            'error_types': {k: counts.get(k, 0) for k in CATEGORIES if counts.get(k, 0)},
            'b1_subtypes': dict(subs),
        }
        # osworldg's 54 refusal items cannot be scored correct by a model that
        # always clicks; quote the reachable ceiling next to the raw number.
        nref = sum(r['box_type'] == 'refusal' for r in recs)
        if nref:
            entry['refusal_items'] = nref
            entry['ceiling'] = round((n - nref) / n * 100, 2)
            nonref = [r for r in recs if r['box_type'] != 'refusal']
            entry['accuracy_nonrefusal'] = round(
                sum(r['correct'] for r in nonref) / len(nonref) * 100, 2)
        summary['datasets'][ds] = entry

        print(f'{ds:14s} n={n:5d} acc={entry["accuracy"]:6.2f}% errors={nf:5d}  ' +
              ' '.join(f'{k.split("_")[0]}:{v}' for k, v in entry['error_types'].items()))
        if args.verify:
            print(verify_against_metrics(ds, rows, recs))

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\nwrote {out_dir}/summary.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
