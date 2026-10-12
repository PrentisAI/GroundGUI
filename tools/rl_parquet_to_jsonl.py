#!/usr/bin/env python3
"""Convert the RL corpus from its Parquet release (screenshots embedded) into the JSONL + image
files layout rl/ trains on. scripts/rloo.sh and scripts/opsd.sh call this automatically when
DATASET points at Parquet; it can also be run by hand:

    python tools/rl_parquet_to_jsonl.py <parquet dir or file> <out_dir>

<out_dir>/images/<path> receives every screenshot (the relative path stored in the Parquet row;
written once, existing files are kept, so an interrupted run can be restarted) and
<out_dir>/train.jsonl one row per Parquet row, in order:
    {"solution": ..., "images": "<absolute image path>", "messages": [...],
     "additional_paras": "<json string>", "sample_id": ...}
train.jsonl is written to a temporary name and renamed at the end, so its presence means the
conversion finished.
"""
import glob
import json
import os
import sys

import pyarrow.parquet as pq


def shards_of(src):
    if os.path.isfile(src):
        return [src]
    for pat in ("*.parquet", "data/*.parquet"):
        found = sorted(glob.glob(os.path.join(src, pat)))
        if found:
            return found
    sys.exit(f"no .parquet files under {src}")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = sys.argv[1], os.path.abspath(sys.argv[2])
    img_root = os.path.join(out, "images")
    os.makedirs(img_root, exist_ok=True)
    final, tmp = os.path.join(out, "train.jsonl"), os.path.join(out, "train.jsonl.partial")
    n_rows = n_new = 0
    with open(tmp, "w", encoding="utf-8") as fo:
        for shard in shards_of(src):
            pf = pq.ParquetFile(shard)
            for g in range(pf.num_row_groups):
                for row in pf.read_row_group(g).to_pylist():
                    paths = []
                    for im in row["images"]:
                        rel = os.path.normpath(im["path"])
                        if rel.startswith("..") or os.path.isabs(rel):
                            sys.exit(f"unsafe image path in {shard}: {im['path']!r}")
                        dst = os.path.join(img_root, rel)
                        if not os.path.exists(dst):
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            with open(dst + ".partial", "wb") as fh:
                                fh.write(im["bytes"])
                            os.replace(dst + ".partial", dst)
                            n_new += 1
                        paths.append(dst)
                    rec = {"solution": row["solution"],
                           "images": paths[0] if len(paths) == 1 else paths,
                           "messages": row["messages"],
                           "additional_paras": row["additional_paras"],
                           "sample_id": row["sample_id"]}
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_rows += 1
            print(f"  {os.path.basename(shard)}: {n_rows} rows so far", file=sys.stderr)
    os.replace(tmp, final)
    print(f"  {n_rows} rows -> {final} ({n_new} images written under {img_root})", file=sys.stderr)


if __name__ == "__main__":
    main()
