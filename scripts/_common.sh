# Shared helpers for scripts/*.sh. Sourced, not executed.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }

need() {  # need VAR "hint": fail unless the environment variable is set
  [ -n "${!1:-}" ] || die "set $1=$2"
}

abspath() { python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"; }

# render_meta <meta.json> <out.json>: substitute ${DATA_ROOT} (required if the meta uses it), and
# make relative root / annotation paths absolute under DATA_ROOT (e.g. ScreenRef-Annotations' meta/)
render_meta() {
  local src="$1" dst="$2"
  [ -f "$src" ] || die "meta not found: $src"
  if grep -q '\${DATA_ROOT}' "$src"; then
    need DATA_ROOT "<directory that holds the training data>"
  fi
  python3 - "$src" "$dst" <<'PY'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
root = os.environ.get("DATA_ROOT", "").rstrip("/")
text = open(src).read().replace("${DATA_ROOT}", root)
meta = json.loads(text)
rel = [(k, f) for k, v in meta.items() if isinstance(v, dict) for f in ("root", "annotation")
       if isinstance(v.get(f), str) and v[f] and not os.path.isabs(v[f]) and "://" not in v[f]]
if rel:
    if not root:
        sys.exit(f"ERROR: {src} has relative paths; set DATA_ROOT=<the directory they are relative to>")
    for k, f in rel:
        meta[k][f] = os.path.join(root, meta[k][f])
    text = json.dumps(meta, indent=1, ensure_ascii=False) + "\n"
open(dst, "w").write(text)
PY
}

# screenref_unpack: if SCREENREF_PARQUET is set (the downloaded ScreenRef screenshots: the directory
# holding data/train-*.parquet, or data/ itself), unpack them once into $DATA_ROOT/images/ with the
# parquet_to_tree.py that ships in ScreenRef-Annotations (DATA_ROOT). Re-runs when the shard count
# changes; files already present are kept.
screenref_unpack() {
  [ -n "${SCREENREF_PARQUET:-}" ] || return 0
  need DATA_ROOT "<the ScreenRef-Annotations directory>"
  local shards="$SCREENREF_PARQUET"; [ -d "$shards/data" ] && shards="$shards/data"
  local n; n=$(ls "$shards"/train-*.parquet 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || die "no train-*.parquet under $shards"
  local mark="$DATA_ROOT/images/.unpacked_shards"
  [ "$(cat "$mark" 2>/dev/null)" = "$n" ] && return 0
  [ -f "$DATA_ROOT/scripts/parquet_to_tree.py" ] \
    || die "DATA_ROOT must be the ScreenRef-Annotations directory (no scripts/parquet_to_tree.py)"
  echo "unpacking $n ScreenRef shards from $shards into $DATA_ROOT/images (once)" >&2
  "$CONDA_ENV/bin/python" "$DATA_ROOT/scripts/parquet_to_tree.py" --shards "$shards" \
    --meta "$DATA_ROOT/meta/meta_full.json" --out "$DATA_ROOT" >&2 || die "parquet_to_tree.py failed"
  echo "$n" > "$mark"
}

# ckpt_format <dir>: prints "tf5" for a checkpoint saved by transformers 5.x (sft/), else "tf4"
ckpt_format() {
  python3 - "$1" <<'PY'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
print("tf5" if str(c.get("transformers_version", "")).startswith("5") else "tf4")
PY
}

# rl_init <checkpoint>: prints a checkpoint the RL stage can load. An SFT checkpoint
# (transformers 5.x) is mirrored once into rl/models/<run>-<checkpoint>-tf4 by prepare_ckpt.py.
rl_init() {
  local src; src="$(abspath "$1")"
  [ -f "$src/config.json" ] || die "not a checkpoint directory: $src"
  if [ "$(ckpt_format "$src")" = tf4 ]; then echo "$src"; return; fi
  local dst="$REPO/rl/models/$(basename "$(dirname "$src")")-$(basename "$src")-tf4"
  if [ ! -f "$dst/config.json" ]; then
    "$ENV_BIN/python" "$REPO/rl/Grounding_scripts/prepare_ckpt.py" --src "$src" --dst "$dst" >&2 \
      || die "prepare_ckpt.py failed for $src"
  fi
  echo "$dst"
}

# rl_dataset <path>: prints the RL corpus .jsonl to train on. A Parquet release (a .parquet file,
# or a directory holding *.parquet or data/*.parquet, e.g. the downloaded RL dataset) is converted
# once into ${RL_DATA_DIR:-rl/data}/<name>/train.jsonl + images/ by tools/rl_parquet_to_jsonl.py.
rl_dataset() {
  local src; src="$(abspath "$1")"
  [ -e "$src" ] || die "dataset not found: $src"
  case "$src" in *.jsonl) echo "$src"; return ;; esac
  local name; name="$(basename "${src%.parquet}")"
  [ "$name" = data ] && name="$(basename "$(dirname "$src")")"
  local dst="${RL_DATA_DIR:-$REPO/rl/data}/$name"
  if [ ! -f "$dst/train.jsonl" ]; then
    echo "converting $src -> $dst (Parquet -> JSONL + images, once)" >&2
    "$ENV_BIN/python" "$REPO/tools/rl_parquet_to_jsonl.py" "$src" "$dst" >&2 \
      || die "tools/rl_parquet_to_jsonl.py failed for $src"
  fi
  echo "$dst/train.jsonl"
}

# run_arm <arm> <run_name>: train one RL arm on this machine -- rl/'s submit.sh with LOCAL=1 runs
# the same job script in the foreground; it stops after training.
run_arm() {
  LOCAL=1 bash "$REPO/rl/Grounding_scripts/opsd_rloo/submit.sh" "$1" "$2"
}
