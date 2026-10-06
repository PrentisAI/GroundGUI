#!/usr/bin/env python3
"""Build the ground_benchmark/*.jsonl files that Grounding_scripts/test_6gpu.sh expects.

The GUI-SD authors never released these files, nor the `os_worldg.py` /
`MMbench.py` converters their metric code refers to.  This script rebuilds them
from the upstream benchmark releases under datasets/gui_eval/.

Output schema (one JSON object per line) mirrors the GUI-SD training file
(gui-sd.jsonl) exactly, minus the assistant turn:

    {
      "solution":         {"name": ..., "arguments": {"action": "click",
                                                      "coordinate": <ground truth>}},
      "images":           "<absolute path to the screenshot>",
      "messages":         [{"role": "system", ...}, {"role": "user", ...}],
      "additional_paras": "<a JSON *string*, not an object>",
      "sample_id":        <int>
    }

`swift infer --remove_unused_columns false` carries `solution` and
`additional_paras` through to the result jsonl, where
swift/metrics/*_metric.py read them back.  `additional_paras` must be a string
because every metric does `json.loads(data['additional_paras'])`.

Ground-truth conventions, per metric (these differ between benchmarks and are
the whole reason this script exists):

    screenspotv2   solution.coordinate = xyxy pixels    (raw is xywh)
    screenspotpro  solution.coordinate = xyxy pixels    (raw is already xyxy)
    mmbench        solution.coordinate = xyxy pixels    (raw is xyxy normalised to [0,1])
    osworldg[_r]   bbox    -> xyxy pixels               (raw is xywh)
                   polygon -> flat [x0,y0,x1,y1,...] pixels, verbatim
                   refusal -> verbatim

Usage:
    python Grounding_scripts/build_ground_benchmark.py                  # all five
    python Grounding_scripts/build_ground_benchmark.py -d screenspotv2
    python Grounding_scripts/build_ground_benchmark.py -d screenspotv2 --limit 32 \
        --suffix _smoke                                              # smoke subset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from glob import glob

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required: pip install pillow")

# Pillow refuses to open the larger ScreenSpot-Pro screenshots otherwise.
Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_EVAL_ROOT = os.path.abspath(
    os.path.join(REPO, '..', 'datasets', 'gui_eval'))
DEFAULT_OUT_DIR = os.path.join(REPO, 'ground_benchmark')

# Verbatim from the GUI-SD training file (gui-sd.jsonl).  Keep byte-identical to the
# training prompt: the model was distilled to answer *this* prompt, and
# swift/custom_utils/format_func.py:extract_action only parses the <tool_call>
# wrapper it induces.
SYSTEM_PROMPT = (
    'You may call one or more functions to assist with the user query.\n'
    'You are provided with function signatures within <tools> ... </tools> XML tags:\n'
    '<tools>\n'
    '{"name": "computer_use", "description": "Use a mouse to interact with a computer.",'
    ' "notes": "Click with the cursor tip centered on targets; avoid edges unless asked.'
    ' Do not use other tools (type, key, scroll, left_click_drag). Only left_click are'
    ' allowed.", "parameters": {"type": "object", "required": ["action"], "properties":'
    ' {"action": {"type": "string", "enum": ["left_click"],"description": "The action to'
    ' perform."}, "coordinate": {"type": "array", "description": "(x, y): pixels from'
    ' left/top. Required for action=left_click."}}}}\n'
    '</tools>\n\n'
    'For each function call, return a JSON object with function name and arguments'
    ' within <tool_call> ... </tool_call> XML tags:\n'
    '<tool_call>\n'
    '{"name": "<function-name>", "arguments": <args-json-object>}\n'
    '</tool_call>'
)

# The metrics never read solution['name'] or solution['arguments']['action'] --
# only ['arguments']['coordinate'].  Mirrored from the training file anyway so a
# ground_benchmark line is indistinguishable from a training line.
SOLUTION_NAME = 'mobile_use'
SOLUTION_ACTION = 'click'

_SIZE_CACHE: dict[str, list[int]] = {}


def image_size(path: str) -> list[int]:
    """[width, height], read from the file header only (no decode)."""
    if path not in _SIZE_CACHE:
        with Image.open(path) as im:
            _SIZE_CACHE[path] = [im.width, im.height]
    return _SIZE_CACHE[path]


class Builder:
    """Accumulates records and the validation problems found while building."""

    def __init__(self, name: str):
        self.name = name
        self.rows: list[dict] = []
        self.problems: list[str] = []
        self.buckets: Counter = Counter()

    def warn(self, msg: str) -> None:
        self.problems.append(msg)

    def add(self, *, image: str, instruction: str, coordinate, paras: dict,
            bucket: tuple) -> None:
        self.rows.append({
            'solution': {
                'name': SOLUTION_NAME,
                'arguments': {'action': SOLUTION_ACTION, 'coordinate': coordinate},
            },
            'images': image,
            'messages': [
                {'content': SYSTEM_PROMPT, 'role': 'system'},
                {'content': instruction, 'role': 'user'},
            ],
            # Every metric does json.loads() on this, so it must be a string.
            'additional_paras': json.dumps(paras, ensure_ascii=False),
            'sample_id': len(self.rows),
        })
        self.buckets[bucket] += 1


def check_image(b: Builder, path: str, declared: list | None, ref: str):
    """Resolve an image, cross-checking any size the annotation declares.

    Returns the real [w, h], or None if the file is missing.  A declared size
    that disagrees with the file silently destroys scoring (every metric scales
    the prediction by image_size), so we always trust the file and shout.
    """
    if not os.path.exists(path):
        b.warn(f'{ref}: missing image {path}')
        return None
    real = image_size(path)
    if declared is not None and list(declared) != real:
        b.warn(f'{ref}: declared image_size {list(declared)} != actual {real}; using actual')
    return real


def check_box(b: Builder, box, size, ref: str) -> None:
    """Sanity-check a final xyxy pixel box against the image bounds."""
    x1, y1, x2, y2 = box
    if not (x2 > x1 and y2 > y1):
        b.warn(f'{ref}: degenerate box {box} (x2<=x1 or y2<=y1)')
    if x1 < 0 or y1 < 0 or x2 > size[0] + 1 or y2 > size[1] + 1:
        b.warn(f'{ref}: box {box} out of bounds for image {size}')


# --------------------------------------------------------------------------
# ScreenSpot-v2
#   raw bbox is [x, y, w, h]; platform comes from the filename, not data_source
#   (screenspotv2_metric.py only tabulates platform in mobile/desktop/web and
#   data_type in text/icon).  No image_size in the annotations -> read from file.
# --------------------------------------------------------------------------
def build_screenspotv2(root: str) -> Builder:
    b = Builder('screenspotv2')
    base = os.path.join(root, 'ScreenSpot-v2')
    img_dir = os.path.join(base, 'screenspotv2_image')

    for platform in ('mobile', 'desktop', 'web'):
        anno = os.path.join(base, f'screenspot_{platform}_v2.json')
        if not os.path.exists(anno):
            b.warn(f'missing annotation file {anno}')
            continue
        with open(anno) as f:
            records = json.load(f)

        for i, r in enumerate(records):
            ref = f'{platform}[{i}]'
            path = os.path.join(img_dir, r['img_filename'])
            size = check_image(b, path, None, ref)
            if size is None:
                continue

            x, y, w, h = r['bbox']
            box = [x, y, x + w, y + h]
            check_box(b, box, size, ref)

            data_type = r.get('data_type', 'unknown')
            if data_type not in ('text', 'icon'):
                b.warn(f'{ref}: data_type {data_type!r} outside the metric buckets')

            b.add(
                image=path,
                instruction=r['instruction'],
                coordinate=box,
                paras={
                    'image_size': size,
                    'platform': platform,
                    'data_type': data_type,
                    'data_source': r.get('data_source', 'unknown'),
                    'img_filename': r['img_filename'],
                },
                bucket=(platform, data_type),
            )
    return b


# --------------------------------------------------------------------------
# ScreenSpot-Pro
#   raw bbox is already [x1, y1, x2, y2]; img_size is declared.
#   screenspotpro_metric.py indexes total_eval[para['group']][para['ui_type']]
#   with six hardcoded groups -- an unexpected group raises KeyError, so we fail
#   loudly here instead.
# --------------------------------------------------------------------------
SSPRO_GROUPS = {'Dev', 'Creative', 'CAD', 'Scientific', 'Office', 'OS'}


def build_screenspotpro(root: str) -> Builder:
    b = Builder('screenspotpro')
    base = os.path.join(root, 'ScreenSpot-Pro')
    img_dir = os.path.join(base, 'images')

    for anno in sorted(glob(os.path.join(base, 'annotations', '*.json'))):
        stem = os.path.basename(anno)
        with open(anno) as f:
            records = json.load(f)

        for i, r in enumerate(records):
            ref = f'{stem}[{i}]'
            path = os.path.join(img_dir, r['img_filename'])
            size = check_image(b, path, r.get('img_size'), ref)
            if size is None:
                continue

            box = list(r['bbox'])
            check_box(b, box, size, ref)

            group = r.get('group', 'unknown')
            ui_type = r.get('ui_type', 'unknown')
            if group not in SSPRO_GROUPS:
                b.warn(f'{ref}: group {group!r} not in the metric\'s six groups '
                       '(screenspotpro_metric.py will KeyError)')
            if ui_type not in ('text', 'icon'):
                b.warn(f'{ref}: ui_type {ui_type!r} outside the metric buckets')

            b.add(
                image=path,
                instruction=r['instruction'],
                coordinate=box,
                paras={
                    'image_size': size,
                    'group': group,
                    'ui_type': ui_type,
                    'platform': r.get('platform', 'unknown'),
                    'application': r.get('application', 'unknown'),
                    'id': r.get('id', ref),
                },
                bucket=(group, ui_type),
            )
    return b


# --------------------------------------------------------------------------
# MMBench-GUI  (L2 element grounding)
#   raw bbox is xyxy normalised to [0, 1]; mmbench_metric.py re-normalises the
#   stored box by image_size, so we must store *pixels*.
# --------------------------------------------------------------------------
def build_mmbench(root: str) -> Builder:
    b = Builder('mmbench')
    base = os.path.join(root, 'MMBench-GUI')
    anno = os.path.join(base, 'L2_annotations.json')
    with open(anno) as f:
        records = json.load(f)

    for i, r in enumerate(records):
        ref = f'L2[{i}] idx={r.get("index")}'
        platform = r.get('platform', 'unknown')
        path = os.path.join(base, 'offline_images', platform, r['image_path'])
        size = check_image(b, path, r.get('image_size'), ref)
        if size is None:
            continue

        nb = r['bbox']
        if max(nb) > 1.0 + 1e-6:
            b.warn(f'{ref}: bbox {nb} is not normalised to [0,1]')
        box = [nb[0] * size[0], nb[1] * size[1], nb[2] * size[0], nb[3] * size[1]]
        check_box(b, box, size, ref)

        data_type = r.get('data_type', 'unknown')
        grounding_type = r.get('grounding_type', 'basic')
        if data_type not in ('text', 'icon'):
            b.warn(f'{ref}: data_type {data_type!r} outside the metric buckets')

        b.add(
            image=path,
            instruction=r['instruction'],
            coordinate=box,
            paras={
                'id': r.get('index', i),
                'app_name': r.get('app_name', 'unknown'),
                'image_size': size,
                'data_type': data_type,
                'platform': platform,
                'grounding_type': grounding_type,
            },
            bucket=(grounding_type, platform, data_type),
        )
    return b


# --------------------------------------------------------------------------
# OSWorld-G  (and the refined variant)
#   osworldg_metric.py branches on additional_paras['box_type']:
#     bbox    -> _is_point_in_rectangle(pred, coordinate)  => coordinate is xyxy
#                (raw box_coordinates is xywh, so we convert)
#     polygon -> _is_point_in_polygon(pred, coordinate)    => flat pixel list, verbatim
#     refusal -> handled without the coordinate            => verbatim
# --------------------------------------------------------------------------
# classification_result.json uses snake_case keys; osworldg_metric.py prints a
# table ordered by these display names, and anything else falls into its
# "other categories" tail.
OSWG_CATEGORY_NAMES = {
    'text_matching': 'Text Matching',
    'element_recognition': 'Element Recognition',
    'layout_understanding': 'Layout Understanding',
    'fine_grained_manipulation': 'Fine-grained-Manipulation',
    'refusal': 'Refusal',
}
# Priority for collapsing the multi-label classification to the single string the
# metric can bucket on.  Refusal first because it is a disjoint box_type.
OSWG_CATEGORY_PRIORITY = ['refusal', 'text_matching', 'element_recognition',
                          'layout_understanding', 'fine_grained_manipulation']


def _load_osworldg_categories(base: str, b: Builder) -> dict:
    """id -> list of category display names.

    The upstream classification is MULTI-LABEL: 1047 labels over 564 records.
    osworldg_metric.py does `category_eval[para['category']].append(...)`, so it
    can only bucket on a single hashable value -- a list would raise
    TypeError: unhashable.  We therefore emit a single primary `category` plus
    the full `categories` list, and the per-category table will differ from the
    official multi-label breakdown.  Total Acc is unaffected.
    """
    path = os.path.join(base, 'benchmark', 'classification_result.json')
    if not os.path.exists(path):
        b.warn('classification_result.json absent; category buckets collapse to "unknown"')
        return {}
    with open(path) as f:
        raw = json.load(f)

    classified = raw.get('classified', raw) if isinstance(raw, dict) else {}
    table: dict = defaultdict(list)
    for key in OSWG_CATEGORY_PRIORITY:
        for entry in classified.get(key, []):
            _id = entry['id'] if isinstance(entry, dict) else entry
            table[_id].append(OSWG_CATEGORY_NAMES.get(key, key))

    unclassified = raw.get('unclassified', []) if isinstance(raw, dict) else []
    if unclassified:
        b.warn(f'{len(unclassified)} records are unclassified; their category is "unknown"')
    if not table:
        b.warn('classification_result.json present but no id->category join found; '
               'category buckets collapse to "unknown"')
    return table


def build_osworldg(root: str, refined: bool) -> Builder:
    b = Builder('osworldg_r' if refined else 'osworldg')
    base = os.path.join(root, 'OSWorld-G')
    fname = 'OSWorld-G_refined.json' if refined else 'OSWorld-G.json'
    anno = os.path.join(base, 'benchmark', fname)
    img_dir = os.path.join(base, 'benchmark', 'images')

    categories = _load_osworldg_categories(base, b)

    with open(anno) as f:
        records = json.load(f)

    for i, r in enumerate(records):
        ref = f'{fname}[{i}] id={r.get("id")}'
        path = os.path.join(img_dir, r['image_path'])
        size = check_image(b, path, r.get('image_size'), ref)
        if size is None:
            continue

        box_type = r.get('box_type', 'unknown')
        coords = r.get('box_coordinates')

        if box_type == 'bbox':
            x, y, w, h = coords
            coordinate = [x, y, x + w, y + h]
            check_box(b, coordinate, size, ref)
        elif box_type == 'polygon':
            coordinate = list(coords)
            if len(coordinate) % 2 != 0 or len(coordinate) < 6:
                b.warn(f'{ref}: polygon has {len(coordinate)} values (need an even '
                       'count >= 6 for the ray-casting test)')
        elif box_type == 'refusal':
            # Scored by osworldg_metric.py as "both predicted coords < 0"; the
            # stored coordinate is never read. Kept verbatim ([0,0,0,0] upstream).
            coordinate = list(coords)
        else:
            b.warn(f'{ref}: unknown box_type {box_type!r}; the metric treats it as wrong')
            coordinate = list(coords)

        # One jsonl mixes 4-element bbox, 10-42 element polygon and refusal rows
        # in a single column. pyarrow infers one type for it, so keep every value
        # a float -- an int64/double mix makes load_dataset('json') throw.
        coordinate = [float(v) for v in coordinate]

        cats = categories.get(r.get('id'), [])
        b.add(
            image=path,
            instruction=r['instruction'],
            coordinate=coordinate,
            paras={
                'id': r.get('id', ref),
                'GUI_types': r.get('GUI_types', ['unknown']),
                'image_size': size,
                'box_type': box_type,
                # Single string: the metric uses it as a dict key.
                'category': cats[0] if cats else 'unknown',
                # Full multi-label set, for a metric that can consume it.
                'categories': cats or ['unknown'],
            },
            bucket=(box_type,),
        )
    return b


BUILDERS = {
    'screenspotv2': lambda root: build_screenspotv2(root),
    'screenspotpro': lambda root: build_screenspotpro(root),
    'mmbench': lambda root: build_mmbench(root),
    'osworldg': lambda root: build_osworldg(root, refined=False),
    'osworldg_r': lambda root: build_osworldg(root, refined=True),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-d', '--datasets', nargs='+', default=list(BUILDERS),
                   choices=list(BUILDERS))
    p.add_argument('--eval-root', default=DEFAULT_EVAL_ROOT,
                   help=f'default: {DEFAULT_EVAL_ROOT}')
    p.add_argument('--out-dir', default=DEFAULT_OUT_DIR,
                   help=f'default: {DEFAULT_OUT_DIR}')
    p.add_argument('--limit', type=int, default=0,
                   help='keep only the first N rows (evenly sampled across buckets) '
                        'for a smoke test; 0 = all')
    p.add_argument('--suffix', default='',
                   help='append to the output filename, e.g. --suffix _smoke')
    args = p.parse_args()

    if not os.path.isdir(args.eval_root):
        return f'eval root not found: {args.eval_root}'
    os.makedirs(args.out_dir, exist_ok=True)

    exit_code = 0
    for name in args.datasets:
        print(f'\n=== {name} ===', flush=True)
        b = BUILDERS[name](args.eval_root)

        rows = b.rows
        if args.limit and len(rows) > args.limit:
            # Round-robin over buckets so a smoke subset still exercises every
            # branch of the metric (icon/text, every platform, every box_type).
            by_bucket = defaultdict(list)
            for row in rows:
                paras = json.loads(row['additional_paras'])
                key = paras.get('box_type') or (
                    paras.get('platform'), paras.get('data_type'),
                    paras.get('group'), paras.get('ui_type'))
                by_bucket[str(key)].append(row)
            picked, queues = [], list(by_bucket.values())
            while len(picked) < args.limit and any(queues):
                for q in queues:
                    if q and len(picked) < args.limit:
                        picked.append(q.pop(0))
            rows = picked
            for j, row in enumerate(rows):
                row['sample_id'] = j

        out = os.path.join(args.out_dir, f'{name}{args.suffix}.jsonl')
        with open(out, 'w') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

        print(f'wrote {len(rows)} rows -> {out}')
        print('buckets:')
        for k, v in sorted(b.buckets.items(), key=lambda kv: str(kv[0])):
            print(f'  {k}: {v}')

        if b.problems:
            exit_code = 1
            print(f'PROBLEMS ({len(b.problems)}):')
            for msg in b.problems[:25]:
                print(f'  ! {msg}')
            if len(b.problems) > 25:
                print(f'  ... and {len(b.problems) - 25} more')
        else:
            print('problems: none')

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
